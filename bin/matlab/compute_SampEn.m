% S = compute_SampEn(x, [m, r, std])
% 
% Compute Sample entropy (SampEn) of x using correlation integrals
% 
% x      : vector (or matrix) containing the univariate (or multivariate) data 
% m      : smaller embedding dimension to consider (default 2)
% r      : radius of balls (resolution) (default 0.5)
% std    : std (default 1) 
%
% S, the returned value, is the Sample Entropy C_r(m+1)-C_r(m)
%
% v2013-04-13
%