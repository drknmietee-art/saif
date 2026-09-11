function setup_paths()
%SETUP_PATHS  Put every source folder of this repository on the MATLAB path.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root,'src')));
addpath(fullfile(root,'config'));
addpath(fullfile(root,'simulink'));
addpath(fullfile(root,'sim'));
fprintf('humanoid-asr paths added (root: %s)\n', root);
end
