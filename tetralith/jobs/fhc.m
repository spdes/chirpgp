% A Fast Algorithm for Maximum Likelihood Estimation of Harmonic Chirp 
% Parameters. Tobias Lindstrøm Jensen, Jesper Kjær Nielsen,
% Jesper Rindom Jensen, Mads Græsbøll Christensen, and Søren Holdt Jensen.  
% IEEE Transactions on Signal Processing, 2017. 
% 
% The implementation is based on 
% https://vbn.aau.dk/en/publications/a-fast-algorithm-for-maximum-likelihood-estimation-of-harmonic-ch
% If the link is invalid, try searching the paper title in
% https://vbn.aau.dk/da/publications/.
clear
clc
close all

addpath('../../others/fhc/util');
addpath('../../others/fhc/frame_analysis');
addpath('../../others/fhc');

c = parcluster('Processes');
fprintf('Running with %d workers. \n', c.NumWorkers);

num_mcs = 100;
mags = {'const', 'damped', 'ou'};

%% Single chirp
for i = 0:num_mcs - 1
    for k = 1:3
        datapath = sprintf('../matlab_data/chirp_mc_%d_mag_%s.mat', ...
            i, mags{k});
        savepath = sprintf('../results/fhc_%s_%d.mat', mags{k}, i);
        batch(c, @fhc_estimator, 3, {datapath, 0.3, 1, 100, savepath});
        fprintf('Single chirp. MC %d with mag %s submitted.\n', i, mags{k});
    end
end

%% Harmonic chirp
num_harmonics = 3;
for i = 0:num_mcs - 1
    for k = 1:3
        datapath = sprintf('../matlab_data/harmonic_chirp_mc_%d_mag_%s.mat', ...
            i, mags{k});
        savepath = sprintf('../results/harmonic_fhc_%s_%d.mat', mags{k}, i);
        batch(c, @fhc_estimator, 3, {datapath, 0.3, num_harmonics, 100, ...
            savepath});
        fprintf('Harmonic chirp. MC %d with mag %s submitted.\n', i, mags{k});
    end
end

%% Wait until all the batch jobs are done
pause(inf)
