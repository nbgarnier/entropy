% [S, std, infos] = compute_Renyi(x, q, [embed_params, [algo_params, [mask]]])
% 
% Compute Renyi entropy of x using nearest neighbors searching
% 
% x            : vector or matrix containing the univariate or multivariate data
% q            : order of the Renyi entropy (real number) (default=2)
% embed_params : struct with embedding parameters as follows:
%   m          : number of points in the past of x to be considered (embedding) (default 1)
%   stride     : distance between 2 points in time for the embedding process (default 1)
% samp_params  : struct with sampling parameters as follows:
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
% S, the returned value, is the estimate of Renyi entropy of order q
%
% v2021-12-21
%
