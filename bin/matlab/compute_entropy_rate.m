% [h, std, infos] = compute_entropy(x, [embed_params, [algo_params, [mask]]])
% 
% Compute Shannon entropy rate of x using nearest neighbors searching
% 
% x            : vector or matrix containing the univariate or multivariate data
% embed_params : struct with scaling/order parameters as follows:
%   m          : order of the entropy rate (default 1)
%   stride     : distance between 2 points in time (default 1)
% algo_params  : struct with algorithm parameters as follows:
%   k          : number of neighbors to consider (default 5)
%   threads    : number of threads to use, which can be either:
%               0  : single-thread algorithm (legacy)
%               >0 : impose the number of threads to use (typically 8 or 16)
%               <1 : audo-adaptative multi-threading (default)
%   method     : method to use, which can be either:
%               0  : H^{(m)}/m  
%               1  : H^{(m+1)} - H^{(m)}  
%               2  : H - MI   (default) 
% mask         : mask to indicate relevant epochs in the signal
%
% h, the returned value, is the estimate of Shannon entropy rate of order m of x
%
% other (optional) returned values are respectively:
% - the std of the estimation
% - the number of errors encountered
% - the number of effective points used
% - the number of independant windows used (if ==1, std is not defined and set to 0)
%   (this last quantity should be equal to stride if using Theiler preconisation)
%
% examples:
% h            = compute_entropy_rate(x)
% [H, std]     = compute_entropy_rate(x, struct('stride',7))
%
% create a struct with parameters for the algorithm and use it:
% algo_params=struct('k',5,'threads',8)
% [h, std]     = compute_entropy_rate(x, struct('stride',7), algo_params)
%
% v2022-03-11
%
