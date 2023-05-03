% [ApEn, SampEn] = compute_KC(x, [m, [stride, [r, [std, [kernel_type]]]]])
% 
% Compute Kolmogorov Complexity estimates of x (using correlation integrals)
% and returns Approximate entropy (ApEn) and SampEn entropy (SampEn) for 1<=k<=m
% 
% x      : vector (or matrix) containing the univariate (or multivariate) data 
% m      : largest embedding dimension (default=2)
% stride : time scale parameter (default=1)
% r      : radius of balls (default 0.25)
% std    : std (default 1) 
% kernel_type : 0 for brickwall kernel (default)
%               1 for Gaussian kernel
%
% The returned variables ApEn and SampEn are vectors containing ApEn and SampEn 
% for embedding dimension k such that 1 <= k <= m
% for k=0, the "simple entropies" H(1) are returned
% for k>0, estimates of "entropy rates" H(k+1)-H(k) are returned
% for k=m, H(m+1)-H(m) is the "entropy rate" of order m
%
% v2020-03-04
%