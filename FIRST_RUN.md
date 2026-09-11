# First run — what to check

**These sources have never been executed.** MATLAB is not available in the environment
where they were written, so every file is unrun and unlinted. The maths was verified
independently (see below), but syntax, toolbox call signatures and Simulink block
parameters have not been exercised. Budget an hour for a first pass.

## Verified without MATLAB

| Check | Result |
|---|---|
| `model_complexity.m` parameter counts | 1,005 / 32,933 / 53,573 — matches Table 7 exactly |
| `model_complexity.m` FLOP counts | 2.0 k / 7.70 M / 12.66 M — matches Table 7 exactly |
| Ratios to the proposed network | 32.8× and 53.3× — matches the text |
| `wilson_ci.m` | z = 1.959964; 651/750 → [0.8419, 0.8904] — matches Section 4.1 |
| Block balance (`function`/`if`/`for`/`end`) | all 35 `.m` files balanced |
| File name vs first function name | all match |

## Run in this order

```matlab
cd humanoid-asr
setup_paths
cfg = paper_config();

% 1. smoke test on synthetic audio -- no recordings needed, ~2 min
run_all('synthetic', true, 'reps', 4, 'stages', {'clean','features','train','evaluate'});

% 2. if that passes, the whole thing
run_all('synthetic', true, 'reps', 10);

% 3. a real corpus anyone can download: Speech Commands v0.02 under data/ (GSC.md)
run_all('corpus', 'gsc', 'maxPerClass', 100, ...
        'stages', {'clean','features','train','evaluate'});   % ~2,400 files
run_all('corpus', 'gsc');                                     % all 14,672

% 4. then the corpus of the paper in data/raw
run_all();
```

Step 3 is the better smoke test now that the public corpus is in `data/`: it is real
audio, it exercises the same code paths the paper's corpus does, and with
`maxPerClass` it finishes in a few minutes. Read `GSC.md` before quoting any number
it produces.

## Likely first-run friction, in order of probability

1. **`Interpreted MATLAB Function` block parameters.** `build_asr_model.m` sets
   `MATLABFcn`, `OutputDimensions`, `Output1D` and `SampleTime`. If R2024b rejects one,
   `get_param(gcb,'DialogParameters')` on a hand-placed block gives the exact names.
2. **`From Workspace` frame width.** The model expects a matrix whose first column is
   time and whose remaining 400 columns are one analysis frame per row. If the block
   complains about dimensions, check `audioFrames` in the base workspace is
   `nSteps × 401`.
3. **`trainNetwork` deprecation.** The CNN and LSTM baselines use
   `trainNetwork` + `classificationLayer`, which R2024b still accepts but warns about.
   Switching to `trainnet` also means dropping `classificationLayer` and passing a loss
   function instead. The classical baselines and the proposed network are unaffected.
4. **`convolution1dLayer` / `globalAveragePooling1dLayer`** need a recent Deep Learning
   Toolbox. If unavailable, the `baselines` stage skips the two sequence models with a
   warning and Tables 8 and 9 come out with the four classical rows.
5. **`net.layers{1}.transferFcn = 'poslin'` with `trainscg`.** Supported, but if the
   training record looks degenerate, try `tansig` (the `patternnet` default) to isolate
   the cause before touching anything else.
6. **`dct` orientation.** `mfcc_frames` calls `dct` on a column vector of log filter
   energies and keeps the first 14. If your MFCCs look wrong, print
   `size(dct(log(fb)))` first.

## Things that are deliberate, not bugs

- **`gsc_config` relaxes three cleaning thresholds.** The published ones reject half
  of Speech Commands, and 280 of its 411 `stop` files, which would leave the reported
  test set unbalanced. Each relaxation is in `cfg.departures` and in `GSC.md`.
- **Speech Commands cleaned files are renamed** to
  `<command>_<speakerhash>_<rep>.wav` and written under `data/clean/train`, `val` or
  `test`. The partition is the folder and the speaker is the name, so `data/clean`
  describes itself and no side table can drift away from it.

- **Causal pooling in Simulink.** Offline, the VAD threshold is 2 % of the maximum frame
  energy of the whole file (Section 3.4). Streaming, the maximum is tracked as frames
  arrive. Documented in `slx_recognize.m`.
- **Pre-emphasis before framing.** `run_simulink_demo` pre-emphasises the stream and then
  frames it, so `slx_mfcc_frame` does not repeat the filter. This is what keeps the model
  and `mfcc_frames` numerically identical.
- **`humanoid_asr_rt.slx` is not in the repository.** It is a build product:
  `build_asr_model.m` writes it. A generated model is diffable, reviewable and cannot
  silently drift from the code; a committed binary cannot.
- **The serial stage of `latency_bench` is stubbed** when no board is attached, and the
  output table says so in its `Description`. Do not report a stubbed run as a hardware
  measurement.
- **Stage 6 of the cleaning pipeline** (double-listener label verification) cannot be
  automated. `clean_corpus` writes a `labelConfirmed` column that defaults to true; set
  it from your listening logs before quoting the 25-file rejection count of Table 5.

## What this cannot do

`run_all('synthetic',true)` will report an accuracy. **It will not be 86.8 %, and it
should not be.** The paper's numbers are properties of 4,728 recordings of ten people.
See `sim/SYNTHETIC.md`.

## Pushing to GitHub

The repository is initialised with one commit and nothing from `data/` or `results/` is
tracked.

```bash
cd "humanoid-asr"
git remote add origin https://github.com/<user>/humanoid-asr.git
git branch -M main
git push -u origin main
```

Before making it public, check the two placeholders that are still in the manuscript and
should match here: the Zenodo DOI in `README.md` and `data/README.md`, and the ethics
approval reference in the paper. Neither is filled in.
