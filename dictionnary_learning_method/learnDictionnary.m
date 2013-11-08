function learnDictionnary(X)
% Each column corresponds to a feature.
% So that, size(X,1)= dim of features
% and size(X,2)= num of features

%% Note: This sample code requires minFunc to run.
%        But you can use softICACost.m with your own optimizer of choice.

addpath ../minFunc_2012/   % this should point to minFunc
                              % http://www.di.ens.fr/~mschmidt/Software/minFunc.html
                              % minFunc 2009 seems to work well
%% Load and configure a training dataset

params.m=size(X,2);                 % num patches
params.n=size(X,1);


m = sqrt(sum(patches.^2));
m(m==0)=1;% avoid 0 issues
X = bsxfunwrap(@rdivide,X,m);

%% Run the optimization

params.lambda = 0.05;
params.numFeatures = 400;
params.epsilon = 1e-5;

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;
%options.display = 'off';
%options.outputFcn = 'showBases';

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01;  % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 
randTheta = randTheta(:);

% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options);   % Use x or xw 

% display result
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');

end
