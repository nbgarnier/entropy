%  [I1, I2, std1, std2, ...] = compute_MI(x, y, [embed_params, [algo_params, [mask]]])
% 
% Compute mutual information in x and y using nearest neighbors search
% 
% x, y         : arrays containing the data (possibly multi-dimensional)
% embed_params : struct with embedding parameters as follows:
%   mx         : number of points in the past of x to be considered (embedding) (default 1)
%   my         : number of points in the past of y to be considered (embedding) (default 1)
%   stride     : distance between 2 points in time for the embedding process (default 1)
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
% mask         : mask to indicate relevant epochs in the signal. Only points with
%               time indices i such that mask(i)!=0 will be considered in the computation.
%
% I1 and I2 are two estimates from two different algorithms (see Kraskov et al).
%
% other (optional) returned values are respectively:
% - the std of the first estimator
% - the std of the second estimator
% - the number of errors encountered
% - the number of effective points used
% - the number of independant windows used (if ==1, std is not defined and set to 0)
%   (this last quantity should be equal to stride if using Theiler preconisation)
%
%
% examples:
% I1       = compute_MI(x,y)
% [I1, I2, std1, std2] = compute_MI(x,y, struct('mx',2,'stride',5))
%
% create a struct with parameters for the algorithm and use it:
% algo_params=struct('k',5,'algo',1+2,threads',8)
% [I1, I2, std1, std2] = compute_MI(x,y struct('mx',2,'stride',5), algo_params)
%
% v2021-12-21
%
