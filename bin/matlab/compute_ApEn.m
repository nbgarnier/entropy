% A = compute_ApEn(x, [m, r, std])
% 
% Compute Approximate entropy (ApEn) of x using correlation integrals
% 
% x      : vector (or matrix) containing the univariate (or multivariate) data 
% m      : smaller embedding dimension (default 2)
% r      : radius of balls (default 0.5)
% std    : std (default 1) 
%
% A, the returned value, is the Approximate Entropy C_r(m+1)-C_r(m)
%
% v2013-04-13
%