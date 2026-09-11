function R = run_all(varargin)
%RUN_ALL  Regenerate every table and figure of the paper from the corpus.
%
%   run_all()                    the corpus of the paper, in data/raw
%   run_all('corpus','gsc')      the public Google Speech Commands v0.02 corpus
%                                unpacked under data/  (see GSC.md)
%   run_all('corpus','gsc','maxPerClass',200)   a quick pass over ~3,500 files
%   run_all('synthetic',true)    generate a synthetic corpus first and run on it
%   run_all('stages',{'clean','features','train','evaluate'})   run a subset
%
%   Stages, and the paper object each one produces:
%     clean      Table 5   cleaning statistics
%     features   Tables 3 and 4 (partition), median frame count of Section 3.6
%     train      Table 6   training configuration and the stopping epoch
%     evaluate   Tables 11 and 12, Figures 12 to 14, Section 4.1 headline
%     baselines  Tables 8 and 9
%     altfeat    Table 10
%     cost       Table 7
%     latency    Table 13
%     loso       leave-two-speakers-out (Section 4.7 further work)
%     simulink   build and exercise humanoid_asr_rt.slx
%
%   Tested on MATLAB R2024b.

p = inputParser;
p.addParameter('synthetic', false, @islogical);
p.addParameter('reps', 10, @isnumeric);
p.addParameter('corpus', 'lab', @(v) any(strcmpi(v, {'lab','gsc'})));
p.addParameter('maxPerClass', Inf, @isnumeric);
p.addParameter('cfg', [], @(v) isempty(v) || isstruct(v));
p.addParameter('stages', {'clean','features','train','evaluate','baselines', ...
                          'altfeat','cost','latency','simulink'}, @iscell);
p.parse(varargin{:});
opt = p.Results;

setup_paths();
if ~isempty(opt.cfg)
    cfg = opt.cfg;
elseif strcmpi(opt.corpus,'gsc')
    cfg = gsc_config();
else
    cfg = paper_config();
end
if isfinite(opt.maxPerClass), cfg.maxPerClass = opt.maxPerClass; end
if strcmpi(cfg.corpus,'gsc') && opt.synthetic
    error('run_all:syntheticGsc', ['The synthetic generator writes the lab corpus ' ...
        'layout into data/raw.\nIt has nothing to do with Speech Commands. Use ' ...
        'one or the other, not both.']);
end
for d = {cfg.dataDir, cfg.resultsDir, cfg.artifactsDir}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end
want = @(s) any(strcmp(opt.stages, s));
R = struct();
t0 = tic;

preflight(cfg, opt, want);
report_departures(cfg);

if opt.synthetic
    fprintf('\n== synthetic corpus ==\n');
    make_synthetic_corpus(cfg, opt.reps);
end

if want('clean')
    fprintf('\n== stage: clean (Section 3.4, Table 5) ==\n');
    R.cleanLog = clean_corpus(cfg);
    R.table05  = report_cleaning(cfg, R.cleanLog);
end

if want('features')
    fprintf('\n== stage: features (Sections 3.5, 3.6, Tables 3 and 4) ==\n');
    needSeq = want('baselines');
    R.D = build_dataset(cfg, cfg.cleanDir, needSeq);
    R.S = split_by_speaker(cfg, R.D);
    R.table03 = partition_table(cfg, R.D, R.S);
    D = R.D; S = R.S; corpus = cfg.corpus;               %#ok<NASGU>
    % The corpus is saved beside the data so that a later stage cannot reload
    % this cache under a different configuration without noticing.
    save(fullfile(cfg.artifactsDir,'dataset.mat'), 'D', 'S', 'corpus', '-v7.3');
end

R = ensure_dataset(R, cfg, want);

if want('train')
    fprintf('\n== stage: train (Section 3.7, Table 6) ==\n');
    [R.net, R.tr, R.info] = train_ann(cfg, R.D, R.S);
    export_runtime(cfg, R.net, R.info);
    f = figure('Color','w'); plotperform(R.tr);
    exportgraphics(f, fullfile(cfg.resultsDir,'fig10_training.png'),'Resolution',600);
    close(f);
end

if want('evaluate')
    fprintf('\n== stage: evaluate (Section 4.1, Tables 11 and 12, Figs 12 to 14) ==\n');
    fprintf('evaluating the ''%s'' corpus: %d utterances, %d in the reported test set\n', ...
            cfg.corpus, numel(R.D.y), numel(R.S.testIdx));
    Z  = (R.D.X - R.info.mu) ./ R.info.sg;
    te = R.S.testIdx;
    R.scores = R.net(Z(:,te));
    [~, yPred] = max(R.scores, [], 1);
    yTrue = R.D.y(te);
    R.report = classification_report(cfg, yTrue, yPred, R.scores);
    make_figures(cfg, R.report, yTrue, R.scores);
    writematrix([yTrue(:) yPred(:) R.scores.'], ...
                fullfile(cfg.resultsDir,'predictions_test.csv'));
    R.yTrue = yTrue; R.yPred = yPred;
end

if want('baselines')
    fprintf('\n== stage: baselines (Tables 8 and 9) ==\n');
    if ~isfield(R,'yTrue'), R.yTrue = R.D.y(R.S.testIdx); end
    [R.table08, preds] = train_baselines(cfg, R.D, R.S, R.net, R.info);
    names = R.table08.Method(2:end);
    R.table09 = mcnemar_paired(cfg, R.yTrue, preds{1}, preds(2:end), names);
end

if want('altfeat')
    fprintf('\n== stage: alternative features (Table 10) ==\n');
    R.table10 = alt_feature_sets(cfg, R.D, R.S);
end

if want('cost')
    fprintf('\n== stage: computational cost (Table 7) ==\n');
    R.table07 = model_complexity(cfg);
end

if want('latency')
    fprintf('\n== stage: latency (Section 3.9, Table 13) ==\n');
    R.table13 = latency_bench(cfg, R.net, R.info);
end

if want('loso')
    fprintf('\n== stage: leave-two-speakers-out (Section 4.7) ==\n');
    [R.ltso, R.ltsoSummary] = loso_cv(cfg, R.D);
end

if want('simulink')
    fprintf('\n== stage: Simulink model ==\n');
    R.model = build_asr_model(cfg);
    try
        R.sim = run_simulink_demo(cfg);
    catch err
        warning('run_all:simulink','Model built but simulation failed: %s', err.message);
    end
end

fprintf('\nrun_all finished in %.1f s. Outputs in %s\n', toc(t0), cfg.resultsDir);
if opt.synthetic
    fprintf(['REMINDER: this run used synthetic audio. The numbers above exercise\n' ...
             'the pipeline; they are not the numbers reported in the paper.\n']);
elseif strcmpi(cfg.corpus,'gsc')
    fprintf(['REMINDER: this run used Google Speech Commands v0.02, not the corpus\n' ...
             'of the paper, and with three cleaning thresholds relaxed. The numbers\n' ...
             'above are a Speech Commands result. They are not a reproduction of\n' ...
             'Section 4.1 and must not be reported as one. See results/departures.txt.\n']);
end
end

% ---------------------------------------------------------------------------
function report_departures(cfg)
%   Print, and record next to the tables, every documented deviation from the
%   published protocol, so that a results folder cannot be read without them.
if ~isfield(cfg,'departures') || isempty(cfg.departures), return, end
fprintf('\n--- departures from the published protocol ---\n');
for k = 1:numel(cfg.departures)
    fprintf('  %s\n', cfg.departures{k});
end
f = fullfile(cfg.resultsDir,'departures.txt');
fid = fopen(f,'w');
if fid > 0
    c = onCleanup(@() fclose(fid));
    fprintf(fid, 'Departures from the protocol published in the paper.\n');
    fprintf(fid, 'Written by run_all on %s.\n\n', char(datetime('now')));
    for k = 1:numel(cfg.departures)
        fprintf(fid, '%s\n', cfg.departures{k});
    end
    clear c
    fprintf('  (also written to %s)\n', f);
end
end

% ---------------------------------------------------------------------------
function preflight(cfg, opt, want)
%   Say what will happen before any stage runs, and stop early with something
%   actionable if there is nothing to work on.
isGsc  = strcmpi(cfg.corpus,'gsc');
nClean = numel(dir(fullfile(cfg.cleanDir, '**', '*.wav')));
fprintf('\n--- humanoid-asr ---\n');
fprintf('root        : %s\n', cfg.root);
fprintf('corpus      : %s\n', corpus_line(cfg));

if isGsc
    nRaw = count_gsc(cfg);
    fprintf('source audio: %d file(s) over %d command folder(s) in %s\n', ...
            nRaw, numel(cfg.gscWords), cfg.gscDir);
else
    nRaw = numel(dir(fullfile(cfg.rawDir, '**', '*.wav')));
    fprintf('raw audio   : %d file(s) in %s\n', nRaw, cfg.rawDir);
end
fprintf('clean audio : %d file(s) in %s\n', nClean, cfg.cleanDir);
fprintf('stages      : %s\n', strjoin(opt.stages, ', '));

if opt.synthetic
    fprintf('mode        : SYNTHETIC (%d repetitions per speaker per command)\n', opt.reps);
    fprintf(['              results will exercise the pipeline; they will NOT\n' ...
             '              match the paper. See sim/SYNTHETIC.md.\n']);
    return
end

if isGsc
    fprintf('mode        : public corpus, Google Speech Commands v0.02\n');
    if isfinite(cfg.maxPerClass)
        fprintf(['              cfg.maxPerClass = %g: training and validation are\n' ...
                 '              subsampled, the test partition is complete.\n'], ...
                 cfg.maxPerClass);
    end
    if want('clean') && nRaw == 0
        error('run_all:noCorpus', ['\n\nNo Speech Commands audio under\n    %s\n\n' ...
          'Unpack the release into data/ so the word folders sit directly in it:\n' ...
          '    curl -LO https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz\n' ...
          '    tar -xzf speech_commands_v0.02.tar.gz -C data\n'], cfg.gscDir);
    end
    if want('clean')
        corpus_census(cfg);
    elseif nClean > 0
        corpus_census(cfg, cfg.cleanDir);
    end
else
    fprintf('mode        : real corpus\n');
    src = cfg.cleanDir;
    if want('clean') && nRaw > 0, src = cfg.rawDir; end
    if nRaw + nClean > 0
        [~, cen] = corpus_census(cfg, src);
        if ~isempty(cen.missingSpeakers) && ~isempty(cen.speakers)
            fprintf(['              NOTE: speakers %s are absent. The split of Table 3\n' ...
                     '              needs S01 to S10; see split_by_speaker for how to\n' ...
                     '              override it for a subset test.\n'], ...
                     mat2str(cen.missingSpeakers));
        end
    end
    if want('clean') && nRaw == 0 && nClean == 0
        error('run_all:noCorpus', ['\n\nThere is no audio to work on.\n\n' ...
          'To run on the public corpus, which anyone can download:\n' ...
          '    run_all(''corpus'', ''gsc'')          (see GSC.md)\n\n' ...
          'To run everything without recordings (about two minutes):\n' ...
          '    run_all(''synthetic'', true, ''reps'', 10)\n\n' ...
          'To run on the corpus of the paper, put the speaker folders under\n' ...
          '    %s\n' ...
          'so that files look like  S01/S01_s1_move_forward_001.wav ,\n' ...
          'then call run_all() again. The corpus is at\n' ...
          '    https://doi.org/10.5281/zenodo.20156841\n' ...
          'and data/CORPUS_LAYOUT.md documents the layout.\n'], cfg.rawDir);
    end
end

needClean = any(cellfun(want, {'features','altfeat','latency','simulink'}));
if needClean && nClean == 0 && ~want('clean')
    error('run_all:noCleanCorpus', ['\n\nStage needs cleaned audio but %s is empty.\n' ...
      'Run the clean stage first:  run_all(%s''stages'', {''clean''})\n'], ...
      cfg.cleanDir, corpus_arg(cfg));
end
if nClean > 0 && ~want('clean')
    check_clean_layout(cfg, isGsc, needClean);
end
end

% ---------------------------------------------------------------------------
function out = ternary_str(c, a, b)
if c, out = a; else, out = b; end
end

% ---------------------------------------------------------------------------
function s = corpus_line(cfg)
if strcmpi(cfg.corpus,'gsc')
    s = sprintf('gsc  (Speech Commands v0.02; %s -> %s)', ...
        strjoin(cfg.gscWords, '/'), strjoin(cfg.commands, '/'));
else
    s = 'lab  (the ten-speaker corpus of the paper)';
end
end

% ---------------------------------------------------------------------------
function s = corpus_arg(cfg)
%   The argument that has to be repeated on the follow-up call, if any.
if strcmpi(cfg.corpus,'gsc')
    s = '''corpus'', ''gsc'', ';
else
    s = '';
end
end

% ---------------------------------------------------------------------------
function n = count_gsc(cfg)
n = 0;
for k = 1:numel(cfg.gscWords)
    n = n + numel(dir(fullfile(cfg.gscDir, cfg.gscWords{k}, '*.wav')));
end
end

% ---------------------------------------------------------------------------
function check_clean_layout(cfg, isGsc, fatal)
%   data/clean is written by CLEAN_CORPUS in one of two layouts: speaker folders
%   for the lab corpus, partition folders for Speech Commands.  Catch the case
%   where it holds the other one now, rather than after BUILD_DATASET has spent
%   an hour on it.
d = dir(cfg.cleanDir);
subs = {d([d.isdir]).name};
subs = subs(~ismember(subs, {'.','..'}));
if isempty(subs), return, end
looksGsc = any(ismember({'train','val','test'}, subs));
looksLab = any(~cellfun(@isempty, regexp(subs, '^S\d+$', 'once')));
% Only a stage that actually reads data/clean has to stop; for the others the
% mismatch is still worth saying out loud, because it usually means the folder
% was overwritten by a later run and the cached artefacts may have gone with it.
if ~fatal && (looksGsc ~= isGsc) && (looksGsc || looksLab)
    warning('run_all:staleCleanLayout', ...
        ['%s holds the %s layout but this run is configured for ''%s''. No stage in ' ...
         'this run reads it, so the run continues, but if data/clean was overwritten ' ...
         'then results/artifacts may have been overwritten with it.'], ...
        cfg.cleanDir, ternary_str(looksGsc,'Speech Commands','lab-corpus'), cfg.corpus);
    return
end
if isGsc && looksLab && ~looksGsc
    error('run_all:wrongCleanLayout', ['\n\n%s holds speaker folders (%s ...),\n' ...
      'which is the lab-corpus layout, but this run is configured for Speech\n' ...
      'Commands. Clean again so the partition folders are rewritten:\n' ...
      '    run_all(''corpus'', ''gsc'', ''stages'', {''clean'',''features''})\n'], ...
      cfg.cleanDir, subs{1});
elseif ~isGsc && looksGsc && ~looksLab
    error('run_all:wrongCleanLayout', ['\n\n%s holds train/val/test folders,\n' ...
      'which is the Speech Commands layout, but this run is configured for the\n' ...
      'corpus of the paper. Clean again:\n' ...
      '    run_all(''stages'', {''clean'',''features''})\n'], cfg.cleanDir);
end
end

% ---------------------------------------------------------------------------
function R = ensure_dataset(R, cfg, want)
%   Let any stage run on its own by loading what an earlier stage cached, and
%   refuse to use a cache that came from a different corpus.  Without that check
%   a run configured for one corpus will silently evaluate another one's cached
%   dataset and print the whole banner, departures included, over the wrong
%   numbers.
needsData = any(cellfun(want, {'train','evaluate','baselines','altfeat','latency','loso'}));
if needsData && ~isfield(R,'D')
    f = fullfile(cfg.artifactsDir,'dataset.mat');
    assert(exist(f,'file')==2, 'run_all:noDataset', ...
        'Stage needs the dataset. Run the features stage first, or run all stages.');
    L = load(f);
    R.D = L.D; R.S = L.S;
    if isfield(L,'corpus'), cached = L.corpus; else, cached = dataset_corpus(L.D); end
    check_corpus(cfg, cached, f, numel(L.D.y));
    fprintf('ensure_dataset: reloaded %s\n', f);
    fprintf('                %d utterances, corpus ''%s'', %d speaker(s)\n', ...
            numel(L.D.y), cached, numel(unique(L.D.speaker)));
end
needsNet = any(cellfun(want, {'evaluate','baselines','latency','simulink'}));
if needsNet && ~isfield(R,'net')
    f = fullfile(cfg.artifactsDir,'asr_runtime.mat');
    if exist(f,'file')
        L = load(f); R.net = L.net; R.info = struct('mu',L.mu,'sg',L.sg);
        if isfield(L,'corpus'), check_corpus(cfg, L.corpus, f, []); end
        fprintf('ensure_dataset: reloaded %s\n', f);
    end
end
end

% ---------------------------------------------------------------------------
function check_corpus(cfg, cached, f, nUtt)
%   The cached artefact has to have come from the corpus this run is configured
%   for.  Anything else is a silent corpus swap, which is the one failure mode
%   that produces a plausible-looking number from the wrong data.
if strcmpi(cached, cfg.corpus), return, end
extra = '';
if ~isempty(nUtt)
    extra = sprintf('It holds %d utterances.\n', nUtt);
end
if strcmpi(cached, 'unknown')
    error('run_all:corpusUnknown', ['\n\n' ...
      'The corpus this cached dataset came from cannot be established:\n    %s\n%s\n' ...
      'It predates both the corpus tag and the partition folders, so there is no\n' ...
      'way to tell whether it matches the ''%s'' configuration of this run.\n' ...
      'Nothing has been computed. Rebuild it:\n' ...
      '    run_all(%s''stages'', {''clean'',''features'',''train''})\n'], ...
      f, extra, cfg.corpus, corpus_arg(cfg));
end
error('run_all:corpusMismatch', ['\n\n' ...
  'The cached dataset was built from the ''%s'' corpus, but this run is\n' ...
  'configured for ''%s''.\n    %s\n%s\n' ...
  'Nothing has been computed. Using it would report that corpus''s numbers\n' ...
  'under this run''s banner, departures included.\n\n' ...
  'Either rebuild the cache for this corpus:\n' ...
  '    run_all(%s''stages'', {''clean'',''features'',''train''})\n' ...
  'or put back the artefacts from the run you meant to use\n' ...
  '(results/artifacts/dataset.mat and asr_runtime.mat) and try again.\n'], ...
  cached, cfg.corpus, f, extra, corpus_arg(cfg));
end

% ---------------------------------------------------------------------------
function tbl = partition_table(cfg, D, S)
%   Tables 3 and 4: composition of the partition, and composition by command.
if strcmpi(cfg.corpus,'gsc')
    tbl = partition_table_gsc(cfg, D, S);
else
    tbl = partition_table_lab(cfg, D, S);
end
disp(tbl);
command_table(cfg, D, S);
end

% ---------------------------------------------------------------------------
function tbl = partition_table_lab(cfg, D, S) %#ok<INUSD>
%   Table 3 as published: one row per speaker, S01 to S10.
sp = unique(D.speaker);
n  = numel(sp);
part = strings(n,1); cleanN = zeros(n,1);
for k = 1:n
    cleanN(k) = sum(D.speaker == sp(k));
    if ismember(sp(k), cfg.trainSpeakers),     part(k) = "Training";
    elseif ismember(sp(k), cfg.valSpeakers),   part(k) = "Validation";
    else,                                      part(k) = "Test (held out)"; end
end
tbl = table(sp(:), cleanN, part, 'VariableNames', {'Speaker','Cleaned','Partition'});
writetable(tbl, fullfile(cfg.resultsDir,'table03_partition.csv'));
end

% ---------------------------------------------------------------------------
function tbl = partition_table_gsc(cfg, D, S)
%   Speech Commands has some 2,600 speakers, so the equivalent of Table 3 is one
%   row per partition carrying its speaker count, not one row per speaker.  The
%   speaker counts are the check that matters: they must not add up to more than
%   the number of distinct speakers in the corpus, or a speaker is in two
%   partitions.  SPLIT_BY_SPEAKER asserts that separately.
names = ["Training"; "Validation"; "Test (held out)"; "Test (reported)"];
idx   = {S.trainIdx, S.valIdx, S.heldIdx, S.testIdx};
nSpk  = zeros(4,1); nUtt = zeros(4,1);
for k = 1:4
    nUtt(k) = numel(idx{k});
    nSpk(k) = numel(unique(D.speaker(idx{k})));
end
tbl = table(names, nSpk, nUtt, 'VariableNames', {'Partition','Speakers','Cleaned'});
writetable(tbl, fullfile(cfg.resultsDir,'table03_partition.csv'));
end

% ---------------------------------------------------------------------------
function command_table(cfg, D, S)
%   Table 4: composition by command.
C = numel(cfg.commands);
cmd = cfg.commands(:); cleaned = zeros(C,1); dev = zeros(C,1);
held = zeros(C,1); rep = zeros(C,1);
for c = 1:C
    cleaned(c) = sum(D.y == c);
    dev(c)     = sum(D.y(S.devIdx)  == c);
    held(c)    = sum(D.y(S.heldIdx) == c);
    rep(c)     = sum(D.y(S.testIdx) == c);
end
t4 = table(cmd, cleaned, dev, held, rep, 'VariableNames', ...
    {'Command','Cleaned','DevelopmentPool','HeldOutPool','ReportedTestSet'});
writetable(t4, fullfile(cfg.resultsDir,'table04_bycommand.csv'));
disp(t4);
end
