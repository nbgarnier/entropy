% [S, infos] = compute_entropy_2d(x, [embed_params, [sampl_params, [algo_params]]])
% 
% Compute Shannon entropy of a matrix (image) x using nearest neighbors searching
% 
% x            : matrix (image) containing the univariate (or multivariate) data 
% embed_params : struct with embedding parameters as follows:
%   mx, my     : number of points to consider in the x- and y- embedding (default 1)
%   stride_x   : distance between 2 points for the x-embedding process (default 1)
%   stride_y   : distance between 2 points for the y-embedding process (default 1)
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
%
% S, the returned value, is the estimate of Shannon entropy of the spatially embedded matrix 
% note that with embedding dimensions (mx, my), the embedded matrix is a matrix of 
% multivariate data of dimension (mx+my-1) (ie, embedding is performed along the two principal axis)
%
% other (optional) returned values are respectively:
% - the std of the estimation
% - the number of errors encountered
% - the number of effective points used
% - the number of independant windows used (if ==1, std is not defined and set to 0)
%   (this last quantity should be equal to stride_x*stride_y if using Theiler=-1 or -3 preconisations)
% - the std of the increments
% - the std of the std of the increments
%
% v2022-05-24
%