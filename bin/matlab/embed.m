function [ xtmp ] = embed(x,m ,stride)
%
% return the signal x embedded m time with a stride egal to stride
% 
%    xembedded = embed(x,m ,stride)
%
%   Inputs:
%     
%       x : a vector
%       m : order of embedding
%      stride : distance between two consecutive points
%
%  Outputs:
%
%       xembedded : embedded signal 
%
%%
% SR, ENS-Lyon  5/2016

si=size(x);
if length(si) >2
    error('The signal must be one dimensional!');
elseif si(1)>1 & si(2)>1
    error('The signal must be one dimensional!');
end

test=0;
if si(1)>1
    x=x';
    test=1;
end
N=length(x);


i=(m-1)*stride+1:1:(N-rem(N,stride));
xtmp=x(i);
for ptemp=2:m
    xtmp=vertcat(xtmp,x(i-(ptemp-1)*stride));
end

if test
xtmp=xtmp';
end

end

