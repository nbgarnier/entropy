function [H, MI, lp, center]=compute_entropyrate_NaN(x,k,p1,p2,stride,N,overlap)
%%
%
%  [H, MI, lp, center]=compute_entropyrate_NaN(x,k,p1,p2,stride,Npts,overlap)
%
%  Compute entropy (H) and Mutual information of a signal with itself (MI(x^(p1),x^(p2))).
%  The entropy rate is then equal to H-MI.
%
%  inputs :
%
%   x       : one dimensional signal 
%   k       : number of neighbors to consider
%   p1      : first embedding 
%   p2      : second embedding 
%   stride  : distance between 2 consecutive points (stride)
%   Npts    : size of the window (default is the  size of x)
%   overlap : overlap of the windows in points (default 0)
%
%  outputs :
% 
%   H      : entropy of the signal x
%   MI     : Mutual Information of x with itself
%   lp     : mean number of point in the computation
%   center : center of the windows
%
%%%
% S.R, ENS Lyon 16/11/2016 
%

lx=size(x);
% verify x
if lx(1) > 1 && lx(2) >1
    error('signal x must be one dimensional');
end
if lx(2) > 1 
    x=x';
end
lx=length(x);
% k 
if k < 2 || rem(k,1)~=0
    error('input k must be integer larger than 1')
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
% overlap
if nargin < 7
    overlap=0;
else
    if overlap < 0 || rem(overlap,1)~=0
        error('input overlap must be positive integer.')
    end
end

% embedding of the signals
tmp = embed(x,p1+p2,stride);
Xtmp=tmp(:,1:p1);
Ytmp=tmp(:,p1+1:p1+p2);
clear tmp
Ntmp=min(size(Xtmp,1));

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
if nargin < 6
    N=Ntmp;
    Nreal=1;
    nooverlap=N;
else
    if N<= overlap
        error('input argument overlap is larger than the size of the signal')
    end
    nooverlap=N-overlap;
    Nreal=floor((Ntmp-overlap)/nooverlap);
    if Nreal < 1
       error('Input N is too large compare to the size of the signals.'); 
    end
end


%Initialisation H
H=zeros(Nreal,1);
MI=zeros(Nreal,1);

%size condition
lp=zeros(Nreal,1);
center=zeros(Nreal,1);

%%
firstIndices=stride*(p1+p2-1);
deb=1;
for ireal=1:Nreal
    % old ones
    %deb=(ireal-1)*N+1;
    %fin=ireal*N;
    fin=deb+N-1;
    X=Xtmp(deb:fin,:); % be careful can be multidimensional if p1>1
    Y=Ytmp(deb:fin,:); % be careful can be multidimensional if p2>1
    mask=maskp(deb:fin); % always one dimension
    
    % computation  
    Htmp=0;
    MItmp=0;
    lptmp=0;
    for is=1:stride
        masktmp=mask(is:stride:end);
        xtmp=X(is:stride:end,:);
        ytmp=Y(is:stride:end,:);
        Ip=find(masktmp);
        lptmp=length(Ip)+lptmp;
        Htmp=compute_entropy(xtmp(Ip,1), k, 1, 1)+Htmp;
        MItmp=compute_MI(xtmp(Ip,:),ytmp(Ip,:),k,1,1,1)+MItmp;
    end
    H(ireal)=Htmp/stride;
    MI(ireal)=MItmp/stride;
    lp(ireal)=lptmp/stride;
    center(ireal)=(fin-deb+1)/2+deb+firstIndices;
    
    deb=deb+nooverlap;
end



  


