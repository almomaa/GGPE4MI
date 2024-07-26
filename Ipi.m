function I = Ipi(x, y, z, varargin)

if nargin<3, z = []; end

inputValidation(x,y,z);

[N,~] = size(x);

p = inputParser;

p.addParameter('K',ceil(log2(N)));             %default K
p.addParameter('uTol',eps);                    %uniqness tolerence
p.addParameter('mu',@(l,f) (l./f)./sum(l./f)); %Probability function
p.addParameter('volm',@(u) prod(u,2));         %Volume Measure
p.addParameter('stallCounter', 30);

p.parse(varargin{:});
options = p.Results;

[piFlag, options] = dataGeometryValidation(x,y,z, options);

if ~piFlag
    I = discreteMutualInformation(x,y,z);
    return;
end

if isempty(z)
    I = MIpi(x,y,options);
    if I<0
        options.mu = @(l,f) (f.*exp(-l))./sum(f.*exp(-l));
    end
    stallCounter = 0;
    while I<0 && stallCounter<options.stallCounter
        stallCounter = stallCounter + 1;
        options.K = options.K + 1;
        I = MIpi(x,y,options);
    end
else
    I = CMIpi(x,y,z,options);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                       Mutual Information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = MIpi(x,y, opt)

[~,dimx] = size(x);


[Lxy, Sxy] = uniqueValuePartitionIntersection([x,y],opt);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                    Conditional Mutual Information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = CMIpi(x,y,z, opt)

[Lx, Sx] = uniqueValuePartitionIntersection(x,opt);
[Ly, Sy] = uniqueValuePartitionIntersection(y,opt);
[Lz, Sz] = uniqueValuePartitionIntersection(z,opt);


Sxz = cat(2,Sx, Sz);
Lxz = cat(2,Lx, Lz);

Syz = cat(2,Sy, Sz);
Lyz = cat(2,Ly, Sz);

Sxyz = cat(2,Sx, Sy, Sz);
Lxyz = cat(2,Lx, Ly, Lz);

% js = macro states (Joint Space)
[~,ia,ic] = unique(Sxyz,"rows");
jsL = Lxyz(ia,:); jsL(~jsL) = eps;
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
%                         Data Geometry Validation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [C, options] = dataGeometryValidation(x,y,z, options)
% % C = false: Data is not sutable for geometric estimation
% % C = true: use PI estimator for the data
% 
% if ~isempty(z)
%     N = zeros(16,3);
%     for i=1:16
%         N(i,1) = size(unique(round(x,i),"rows"),1);
%         N(i,2) = size(unique(round(y,i),"rows"),1);
%         N(i,3) = size(unique(round(z,i),"rows"),1);
%     end
% else
%     N = zeros(16,2);
%     for i=1:16
%         N(i,1) = size(unique(round(x,i),"rows"),1);
%         N(i,2) = size(unique(round(y,i),"rows"),1);
%     end
% end
% 
% STD = std(N,[],1);
% 
% if all(~STD), C = false; return; end
% 
% G = gradient(N')';
% [~,c] = find(~G',1,"first");
% if max(N(c,:))<2^size(x,2), C = false; return; end
% 
% if ~isempty(c)
%     options.uTol = 10^(-c);
% end
% C = true;
% end
function [C, options] = dataGeometryValidation(x, y, z, options)
% Validate data for geometric estimation
% C = false: Data is not suitable for geometric estimation
% C = true: Use PI estimator for the data

% Initialize the number of dimensions based on the presence of z
if ~isempty(z)
    numDims = 3;
    data = {x, y, z};
else
    numDims = 2;
    data = {x, y};
end

% Preallocate N matrix to store unique counts
N = zeros(16, numDims);

% Calculate the number of unique rounded values for each dimension
for i = 1:16
    for j = 1:numDims
        N(i, j) = size(unique(round(data{j}, i), 'rows'), 1);
    end
end

% Compute the standard deviation of N
STD = std(N, 0, 1);

% Check if all standard deviations are zero
if all(STD == 0)
    C = false;
    return;
end

% Calculate the gradient and find the first zero gradient
G = gradient(N')';
[~, c] = find(~G', 1, 'first');

% Validate the maximum value in N against the threshold
if max(N(c, :)) < 2^size(x, 2)
    C = false;
    return;
end

% Set options.uTol based on the first non-zero gradient
if ~isempty(c)
    options.uTol = 10^(-c);
end

% Set the output to true indicating the data is suitable
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
%                       Unique value partition intersection
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dL, dS] = uniqueValuePartitionIntersection(x,opt)

u = uniquetol(x,opt.uTol,"ByRows",true);

k = opt.K;
[n,dim] = size(u);


if k>n, k = n; end

t = linspace(0,1,k+1)';

Q = zeros(k+1,dim);
for i=1:dim
    Q(:,i)   = quantile(u(:,i),t);
    Q(1,i)   = min(x(:,i));
    Q(end,i) = max(x(:,i));
end

L = Q(2:end,:)-Q(1:end-1,:);


dS = zeros(size(x,1),dim);
dL = dS;

for i=1:dim
    dS(:,i) = discretize(x(:,i),Q(:,i));
    dL(:,i) = L(dS(:,i),i);
end

end