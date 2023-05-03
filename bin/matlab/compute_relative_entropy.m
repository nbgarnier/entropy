%  [Hr, H, KL] = compute_relative_entropy(x, y, [embed_params, [algo_params, [mask]]])
% 
% Compute relative entropy from datasets x and y, using nearest neighbors
%
% x,y          : vectors or matrices containing the univariate or multivariate data
%                 x is drawn from a PDF f, and y from a PDF g
% embed_params : struct with embedding parameters as follows:
%   px         : number of points in the past of x to be considered (embedding) (default 1)
%   py         : number of points in the past of y to be considered (embedding) (default 1)
%   stride     : distance between 2 points in time for the embedding process (default 1)
% algo_params  : struct with algorithm parameters as follows:
%   k          : number of neighbors to consider (default 5)
%   threads    : number of threads to use, which can be either:
%               0  : single-thread algorithm (legacy)
%               >0 : impose the number of threads to use (typically 8 or 16)
%               <1 : audo-adaptative multi-threading (default)
% mask         : mask to indicate relevant epochs in the signal

% returned values:
% Hr     : the relative entropy Hr(f,g) between PDFs f(x) and g(y)
% H      : the entropy of f (estimated from data in x)  [optional]
% KL     : the KL divergence between f and g              [optional]
%
% v2021-12-23 N.B.G.
%

