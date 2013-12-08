addpath(genpath('../minFunc_2012/'))

%% Note: This sample code requires minFunc to run.
%        But you can use softICACost.m with your own optimizer of choice.
%clear all;
addpath ~/work/RNN/minFunc/   % this should point to minFunc
                              % http://www.di.ens.fr/~mschmidt/Software/minFunc.html
                              % minFunc 2009 seems to work well
%% Load and configure a training dataset
global params;
params.m=3000;%20000;                 % num patches
params.patchWidth=16;           % width of a patch
params.n=params.patchWidth^2;   % dimensionality of input
params.n=100;%32;
% load the patch dataset
%load hyv_patches_16.mat

% for best results patches should be whitened (i.e., patches*patches' ~=
% I)
% [patches, mean_patch, V] = preprocess(patches)
%m = sqrt(sum(patches.^2) + (1e-8));
%x = bsxfunwrap(@rdivide,patches,m);

x=Xpieceweise;

%% Run the optimization
%for i=1:12;
 
params.lambda = 0.05;%10^-(2*(-1+i));
%params.lambda=params.lambda*10^i;
params.numFeatures = 100;%2*params.n%2*params.n;%400;
params.epsilon = 1e-5;

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;% PREVIOUSLY, 300
%options.display = 'off';
%options.outputFcn = 'showBases';

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.1;  % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 
randTheta = randTheta(:);
%if(i>1)
randTheta=opttheta(:);
%end
% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost_L1L2(theta, x, params), randTheta, options);   % Use x or xw 
%end


% display result
W = reshape(opttheta, params.numFeatures, params.n);
W=l2rowscaled(W,1);
%display_network(W');


