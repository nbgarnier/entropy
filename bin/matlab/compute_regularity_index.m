% [Delta1, Delta2, std, infos] = compute_regularity_index(x, [embed_params, [algo_params, [mask]]])
% 
% Compute regularity index H(inc)-h of signal x using nearest neighbors searching
%    this index is computed as a mutual information (see article in Entropy (2021)
% 
% x            : vector or matrix containing the univariate or multivariate data
% embed_params : struct with scaling/order parameters as follows:
%   m          : order of the entropy rate and increments (default 1)
%   stride     : distance between 2 points in time (default 1)
% algo_params  : struct with algorithm parameters as follows:
%   k          : number of neighbors to consider (default 5)
%   algo       : algorithm (from Kraskov et al) to use. This value can be either:
%               1  : for algorithm 1 (default)
%               2  : for algorithm 2
%               1+2: for both algorithms (twice slower)
%   threads    : number of threads to use, which can be either:
%               0  : single-thread algorithm (legacy)
%               >0 : impose the number of threads to use (typically 8 or 16)
%               <1 : audo-adaptative multi-threading (default)
% mask         : mask to indicate relevant epochs in the signal
%
% Delta1 and Delta2, the returned values, are estimates of the regularity index Delta
%
% other (optional) returned values are respectively:
% - the std of the first estimator
% - the std of the second estimator
% - the number of errors encountered
% - the number of effective points used
% - the number of independant windows used (if ==1, std is not defined and set to 0)
%   (this last quantity should be equal to stride if using Theiler preconisation)
%
% examples:
% Delta                = compute_regularity_index(x)
% [D1, D2, std1, std2] = compute_regularity_index(x, struct('stride',7))
%
% create a struct with parameters for the algorithm and use it:
% algo_params=struct('k',5,'threads',8,'algo',2)
% [D1, D2, std1, std2] = compute_regularity_index(x, struct('stride',7), algo_params)
%
% v2022-03-11
%
