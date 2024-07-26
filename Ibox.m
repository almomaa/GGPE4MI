function I = Ibox(x,y,z, varargin)

if nargin<3, z = []; end
inputValidation(x,y,z);

[x,y,z] = rescaleDynamics(x,y,z);

[N,dim] = size(x);

p = inputParser;

p.addParameter('K',ceil(log2(N)));             %default K
p.addParameter('b',5); %default base
p.addParameter('uTol',eps);                    %uniqness tolerence
p.addParameter('mu',@(l,f) (l./f)./sum(l./f)); %Probability function
p.addParameter('volm',@(u) prod(u,2));      %Volume Measure
p.addParameter('stallCounter', 30);

p.parse(varargin{:});
options = p.Results;

[piFlag, options] = dataGeometryValidation(x,y,z, options);

options.K = nProductK(dim, options.K, options.b);


if ~piFlag
    I = discreteMutualInformation(x,y,z);
    return;
end

if isempty(z)
    I = MIbox(x,y,options);
    if I<0
        options.mu = @(l,f) (f.*exp(-l))./sum(f.*exp(-l));
    end
    stallCounter = 0;
    while I<0 && stallCounter<options.stallCounter
        stallCounter = stallCounter + 1;
        options.K = options.K + 1;
        I = MIbox(x,y,options);
    end
else
    I = CMIbox(x,y,z,options);
end

end

function I = MIbox(x,y, opt)

[n,dimx] = size(x);


[L, S] = ndsplit([x;y],[],opt.K,opt.uTol);

Lxy = cat(2,L(1:n,:), L(n+1:end,:));
Sxy = cat(2,S(1:n,:), S(n+1:end,:));

Sx = Sxy(:,1:dimx);
Lx = Lxy(:,1:dimx);

Sy = Sxy(:,dimx+1:end);
Ly = Lxy(:,dimx+1:end);

% js = macro states (Joint Space)
[~,ia,ic] = unique(Sxy,"rows");
jsL = Lxy(ia,:); jsL(~jsL) = eps;
jsF = accumarray(ic,1);


% Marginals

[~,ia,ic] = unique(Sx,"rows");
mLx = Lx(ia,:); mLx(~mLx) = eps;
mFx = accumarray(ic,1);

[~,ia,ic] = unique(Sy,"rows");
mLy = Ly(ia,:); mLy(~mLy) = eps;
mFy = accumarray(ic,1);


Pxy = opt.mu(opt.volm(jsL),jsF);
Px  = opt.mu(opt.volm(mLx),mFx);
Py  = opt.mu(opt.volm(mLy),mFy);


H   = @(p) -dot(p(~~p),log(p(~~p)));


I = (H(Px) + H(Py) - H(Pxy));

end

function I = CMIbox(x,y,z, opt)

ux = unique(x,"rows");
uy = unique(y,"rows");
uz = unique(z,"rows");
uxz = unique([x;z]);

if size(ux,1)<3 || size(uy,1)<3 || size(uz,1)<3 || size(ux,1)==size(uxz,1)
    hxz = dEntropy([x,z]);
    hyz = dEntropy([y,z]);
    hz  = dEntropy(z);
    hxyz = dEntropy([x,y,z]);
    I = hxz + hyz - hxyz - hz;
    return;
end

[n,dimx] = size(x);

[L, S] = ndsplit([x;y;z],[],opt.K, opt.uTol);


L = cat(2,L(1:n,:), L(n+1:2*n,:), L(2*n+1:end,:));
S = cat(2,S(1:n,:), S(n+1:2*n,:), S(2*n+1:end,:));

Sxz = cat(2,S(:,1:dimx), S(:,2*dimx+1:end));
Lxz = cat(2,S(:,1:dimx), S(:,2*dimx+1:end));

Syz = S(:,dimx+1:end);
Lyz = L(:,dimx+1:end);

Sz = S(:,2*dimx+1:end);
Lz = L(:,2*dimx+1:end);

% js = macro states (Joint Space)
[~,ia,ic] = unique(S,"rows");
jsL = L(ia,:);
jsF = accumarray(ic,1);

% Marginals

[~,ia,ic] = unique(Sxz,"rows");
mLx = Lxz(ia,:);
mFx = accumarray(ic,1);

[~,ia,ic] = unique(Syz,"rows");
mLy = Lyz(ia,:);
mFy = accumarray(ic,1);

[~,ia,ic] = unique(Sz,"rows");
mLz = Lz(ia,:);
mFz = accumarray(ic,1);

Pxyz = opt.mu(opt.volm(jsL),jsF);
Pxz  = opt.mu(opt.volm(mLx),mFx);
Pyz  = opt.mu(opt.volm(mLy),mFy);
Pz   = opt.mu(opt.volm(mLz),mFz);


H   = @(p) -dot(p(~~p),log(p(~~p)));


I = (H(Pxz) + H(Pyz) - H(Pxyz)-H(Pz));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   Discrete Mutual Information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = discreteMutualInformation(x,y,z)

if ~isempty(z)
    I = discreteEntropy([x,z]) + discreteEntropy([y,z]) - ...
        discreteEntropy([x,y,z]) - discreteEntropy(z);
else
    I = discreteEntropy(x) + discreteEntropy(y) - ...
        discreteEntropy([x,y]);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   Discrete Entropy
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function E = discreteEntropy(X)
% Entropy estimator of a discrete random variable
%
% Inputs:
%    X: is an n-by-d symbolic matrix (that can be integers or characters)
%     n points in the d-dimensional sample space
 
% Outputs:
%    E: Entropy of X
%
%


[~,~,ic] = unique(X,'rows');
P = accumarray(ic,1)./length(ic);
E = -dot(P,log(P));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         Request Classification
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C, options] = dataGeometryValidation(x,y,z, options)
% C = false: Data is not sutable for geometric estimation
% C = true: use PI estimator for the data

if ~isempty(z)
    N = zeros(16,3);
    for i=1:16
        N(i,1) = size(unique(round(x,i),"rows"),1);
        N(i,2) = size(unique(round(y,i),"rows"),1);
        N(i,3) = size(unique(round(z,i),"rows"),1);
    end
else
    N = zeros(16,2);
    for i=1:16
        N(i,1) = size(unique(round(x,i),"rows"),1);
        N(i,2) = size(unique(round(y,i),"rows"),1);
    end
end

% plot(N)
% pause
STD = std(N,[],1);

if all(~STD), C = false; return; end

G = gradient(N')';
[~,c] = find(~G',1,"first");
if max(N(c,:))<2^size(x,2), C = false; return; end

if ~isempty(c)
    options.uTol = 10^(-c);
end
C = true;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Input Validation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function inputValidation(x,y,z)

if ~isempty(z)
    if ~isequal(size(x),size(y),size(z))
        eid = 'Size:notEqual';
        msg = 'Inputs must have equal size.';
        error(eid,msg)
    end
else
    if ~isequal(size(x),size(y))
        eid = 'Size:notEqual';
        msg = 'Inputs must have equal size.';
        error(eid,msg)
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Rescale Dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function varargout = rescaleDynamics(varargin)

varargout = cell(1,nargin);
for i=1:nargin
    X = varargin{i};
    varargout{i} = reshape(rescale(X(:)),size(X));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Rescale Dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function N = nProductK(n, K, m)

% minimum n numbers that product to K or higher
% numbers are as close as possible to each other
% minimum entry is m

if nargin<3
    m = 2;
end

N = m*ones(1,n);

while prod(N)>n
    m = m-1;
    N = m*ones(1,n);
end


while prod(N)<K
    ix = find(N<N(1),1,"first");
    if ~isempty(ix)
        N(ix) = N(ix)+1;
    else
        N(1) = N(1)+1;
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Rescale Dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L,S] = ndsplit(x,S,K, uTol)
[n,dim] = size(x);

K = min(K,size(unique(x,"rows"),1));

if isempty(S)
    S = ones(n,1);
end
l = S;
for i=1:dim
    for j=1:max(S(:,i))
        l(S(:,i)==j) = ksplit(x(S(:,i)==j,i),K(i),uTol, j);
    end
    S = cat(2,S,l);
end
S = S(:,2:end);

[~,~,ic] = unique(S,"rows");

D = zeros(max(ic),dim);

for i=1:length(D)
    y = x(ic==i,:);
    D(i,:) = max(y,[],1)-min(y,[],1);
end
L=zeros(size(S));
for i=1:dim
    ix = makeSequential(S(:,i));
    L(:,i) = D(ix,i);
end

end


function y = makeSequential(x)
    % Ensure the input is a column vector
    x = x(:);
    
    % Find unique elements and their indices
    [~, ~, y] = unique(x);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Rescale Dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function l = ksplit(x,K,tol,index)

s = linspace(0,1,K+1)';
u = uniquetol(x,tol);
if isscalar(u)
    l = ones(size(x));
else
    t = unique(quantile(u,s));
    t([1 end]) = [min(x); max(x)];
    l = discretize(x,t);
end

l = l + K*(index-1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Rescale Dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = dEntropy(X)

% Entropy estimator of a discrete random variable
%
% Inputs:
%    X: is an n-by-d symbolic matrix (that can be integers or characters)
%     n points in the d-dimensional sample space
 
% Outputs:
%    E: Entropy of X
%
%


[~,~,ic] = unique(X,'rows');
P = accumarray(ic,1)./length(ic);
E = -dot(P,log(P));
end