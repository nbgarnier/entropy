

function [TE1,TE2,lp, center]=computeTE_NaN(x,y,k,p1,p2,stride,lag,N,overlap)
%%
%
%  [TE1,TE2,lp, center]=computeTE_NaN(x,y,k,p1,p2,stride,lag,Npts,overlap)
%
%  Compute entropy (H) and Mutual information of a signal with itself (MI(x^(p1),x^(p2))).
%  The entropy rate is then equal to H-MI.
%
%  inputs :
%
%   x       : one dimensional signal 
%   y       : one dimensional signal (y different of x)
%   k       : number of neighbors to consider
%   p1      : embedding for signal x
%   p2      : embedding for signal y
%   stride  : distance between 2 consecutive points (stride)
%   lag     : distance between present and future point
%   Npts    : size of the window (default is the  size of x)
%   overlap : overlap of the windows in points (default 0)
%
%  outputs :
% 
%   TE1    : transfer entropy from the signal y to x
%   TE2    : transfer entropy from the signal x to y
%   lp     : mean number of point in the computation
%   center : center of the windows
%
%%%
% S.R, ENS Lyon 16/11/2016 

lx=size(x);
ly=size(y);

% verify x
if lx(1) > 1 && lx(2) >1
    error('signal x must be one dimensional');
end
if lx(2) > 1 
    x=x';
end

% verify x
if ly(1) > 1 && ly(2) >1
    error('signal y must be one dimensional');
end
if ly(2) > 1 
    y=y';
end


lx=length(x);
ly=length(y);
if lx ~=ly
    error('signal x and y must have the same size');
end
% k 
if k < 2 || rem(k,1)~=0
    error('input k must be  integer larger than 1')
end
% p1 
if p1 < 1 || rem(p1,1)~=0
    error('input p1 must be positive integer')
end
% p2 
if p2 < 1 || rem(p2,1)~=0
    error('input p2 must be positive integer')
end
% stride
if stride < 1 || rem(stride,1)~=0
    error('input stride must be positive integer')
end
% lag
if lag < 1 || rem(lag,1)~=0
    error('input lag must be positive integer')
end
% overlap
if nargin < 9
    overlap=0;
else
    if overlap < 0 || rem(overlap,1)~=0
        error('input overlap must be positive integer.')
    end
end


% embedding of the signals
Xtmp = embed(x,p1 ,stride);
Ytmp = embed(y,p2 ,stride);
Ntmp=min(size(Xtmp,1),size(Ytmp,1));

% re-sync the signals
dec=stride*abs(p2-p1)+1; 
if p2>p1
   Xtmp=Xtmp(dec:Ntmp,:);
   Ytmp=Ytmp(1:Ntmp,:);
else
   Xtmp=Xtmp(1:Ntmp,:);
   Ytmp=Ytmp(dec:Ntmp,:);
end
Ntmp=min(size(Xtmp,1),size(Ytmp,1));

% mask : find index of NaN
if p1>1
    tmp=sum(Xtmp,2)';
else
    tmp=Xtmp;
end
INaN1=find(isnan(tmp));
if p2>1
    tmp=sum(Ytmp,2)';
else
    tmp=Ytmp;
end
INaN2=find(isnan(tmp));
INaN=union(INaN1,INaN2);
clear INaN1 INaN2 tmp
maskp=ones(lx,1);
maskp(INaN)=0;

% formating data if N < lx
if nargin < 8
    N=Ntmp;
    Nreal=1;
    nooverlap=N;
else
    if N< overlap
        error('input argument overlap is larger than the size of the signal')
    end
    nooverlap=N-overlap;
    Nreal=floor((Ntmp-overlap)/nooverlap);
    if Nreal < 1
       error('Input N is too large compare to the size of the signals.'); 
    end
    
end

%Initialisation 
TE1=zeros(Nreal,1);
TE2=zeros(Nreal,1);
lp=zeros(Nreal,1);
center=zeros(Nreal,1);

%%
% Xtmp(1:3,:)'
% Ytmp(1:3,:)'

firstIndices=stride*(max(p1,p2)-1);
deb=1;
for ireal=1:Nreal
    % old one
    %deb=(ireal-1)*N+1;
    %fin=ireal*N;
    % new one
    fin=deb+N-1;
    
    X=Xtmp(deb:fin,:); % be careful can be multidimensional if p1>1
    Y=Ytmp(deb:fin,:); % be careful can be multidimensional if p2>1
    mask=maskp(deb:fin); % always one dimension
    
    % computation
    TE1tmp=0;
    TE2tmp=0;
    lptmp=0;
    for is=1:stride
        masktmp=mask(is:stride:end);
        xtmp=X(is:stride:end,:);
        ytmp=Y(is:stride:end,:);
        Ip=find(masktmp);
        lptmp=length(Ip)+lptmp;
        
        TE1tmp=compute_TE(xtmp(Ip,:),ytmp(Ip,:),k,1,1,1,lag)+TE1tmp;
        TE2tmp=compute_TE(ytmp(Ip,:),xtmp(Ip,:),k,1,1,1,lag)+TE2tmp;
        
    end
    TE1(ireal)=TE1tmp/stride;
    TE2(ireal)=TE2tmp/stride;
    lp(ireal)=lptmp/stride;
    center(ireal)=(fin-deb+1)/2+deb+firstIndices;
    
    deb=deb+nooverlap;
end



  


