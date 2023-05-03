% [S, std, infos] = compute_entropy(x, [embed_params, [sampl_params, [algo_params, [mask]]]])
% 
% Compute Shannon entropy in x using nearest neighbors searching
% 
% x            : vector or matrix containing the univariate or multivariate data
% embed_params : struct with embedding parameters as follows:
%   m          : number of points in the past of x to be considered (embedding) (default 1)
%   stride     : distance between 2 points in time for the embedding process (default 1)
% samp_params  : struct with s parameters as follows:
%   Theiler    : Theiler scale (>0), or (-1, -2, -3 or -4) for automatic selection
%   N_eff      : number of points to use (>0), or -1 for auto-detect (max)
%   N_real     : number of realizations to use (>0), or (-1, -2) for auto-detect
% algo_params  : struct with algorithm parameters as follows:
%   k          : number of neighbors to consider (default 5)
%   threads    : number of threads to use, which can be either:
%               0  : single-thread algorithm (legacy)
%               >0 : impose the number of threads to use (typically 8 or 16)
%               <1 : audo-adaptative multi-threading (default)
% mask         : mask to indicate relevant epochs in the signal
%
% S, the returned value, is the estimate of Shannon entropy
%
% other (optional) returned values are respectively:
% - the std of the estimation
% - the number of errors encountered
% - the number of effective points used
% - the number of independant windows used (if ==1, std is not defined and set to 0)
%   (this last quantity should be equal to stride if using Theiler preconisation)
%
% examples:
% H        = compute_entropy(x)
% [H, std] = compute_entropy(x, struct('m',2,'stride',5))
%
% create a struct with parameters for the algorithm and use it:
% algo_params=struct('k',5,'threads',8)
% [H, std] = compute_entropy(x, struct('m',2,'stride',5), algo_params)
%
% v2022-05-24
%
