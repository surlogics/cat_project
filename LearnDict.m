function [optDic,cost]=LearnDict(X, lambda, numOut)
% Each column corresponds to a feature.
% So that, size(X,1)= dim of features
% and size(X,2)= num of features

addpath('minFunc_2012/minFunc')%genpath('./rica/'));   % this should point to minFunc
                              % http://www.di.ens.fr/~mschmidt/Software/minFunc.html
                              % minFunc 2009 seems to work well --- Note :
                              % we used minFunc 2012, prying it will work.

% Configuration of parameters for the cost and gradient functions
params.m=size(X,2); % num of data samples
params.n=size(X,1); % dim of feature vector


m = sqrt(sum(X.^2));
m(m==0)=1;% avoid 0 issues
X = bsxfunwrap(@rdivide,X,m);

params.lambda = lambda; %0.05; % Lagrangian constant
params.numFeatures = numOut; %400; % Number of atoms in the dictionary learned
params.epsilon = 1e-5; % Constant to avoid 0 values

% minFunc Parameters
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;

%options.display = 'off'; <-- ??
%options.outputFcn = 'showBases'; <-- ??

% initialize with random weights - original initialization in Rica
randDic = randn(params.numFeatures,params.n)*0.01;  % 1/sqrt(params.n);
randDic = randDic ./ repmat(sqrt(sum(randDic.^2,2)), 1, size(randDic,2)); 
randDic = randDic(:);

% optimize
%[optDic, cost, exitflag] = minFunc( @(dic) softICACost(dic, X, params), randDic, options);   % Use x or xw 
options.GradObj = 'on';
[optDic, cost, exitflag] = fminunc( @(dic) softICACost(dic, X, params),randDic,options);   % Use x or xw 

% display result
optDic = reshape(optDic, params.numFeatures, params.n);
end
