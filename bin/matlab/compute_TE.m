%  [T1, T2, std1, std2, ...] = compute_TE(x, y, [embed_params, [algo_params, [mask]]])
% 
% Compute transfer entropy from y to x using nearest neighbors search
% 
% x, y         : arrays (vectors or matrices) containing the data 
% embed_params : struct with embedding parameters as follows:
%   mx         : number of points in the past of x to be considered (embedding) (default 1)
%   my         : number of points in the past of y to be considered (embedding) (default 1)
%   stride     : distance (e.g., in time) between 2 points for the embedding process (default 1)
%   lag        : distance (e.g., in time) to the "future" point of x (default 1)
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
% T1 and T2 are two estimates from two different algorithms
%
% v2021-12-22, untested
%