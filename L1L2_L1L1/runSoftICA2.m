%% Note: This sample code requires minFunc to run.
%        But you can use softICACost.m with your own optimizer of choice.
%clear all;
addpath ~/Desktop/PhD/myOptimizationConvexL1L2/minFunc/   % this should point to minFunc
                              % http://www.di.ens.fr/~mschmidt/Software/minFunc.html
                              % minFunc 2009 seems to work well
%% Load and configure a training dataset
global params;
params.m=10000;                 % num patches
%params.patchWidth=6;           % width of a patch
%params.n=params.patchWidth^2;   % dimensionality of input
params.n=32;
% load the patch dataset
%load hyv_patches_16.mat

%patches=Xcaltech;
patches=Xpieceweise;%rah2(1:36,1:16000);%Xcaltech;
%patches=Xpieceweise;%(:,1:10000);
% for best results patches should be whitened (i.e., patches*patches' ~=
% I)
% [patches, mean_patch, V] = preprocess(patches)
%patches=bsxfun(@minus,patches,mean(patches));
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
nIteration=5;


for i=3:8%1:nIteration

params.lambda = 10^-(i-1)%0.05/5e3;%0.05;%0.05; METTRE FACTEUR 2 si carré
params.numFeatures = 2*params.n;
params.epsilon = 1e-5;


%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 800;
options.TolX = 1e-5;

%options.display = 'off';
%options.outputFcn = 'showBases';


if i==3
% initialize with random weights
randTheta = log(abs(randn(params.numFeatures,params.n)*0.01));%* 1/sqrt(params.n)));
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 

%randTheta=Wmin(:);
randTheta = randTheta(:);
else
randTheta = (opttheta);
end



% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost4(theta, x, params), randTheta, options);   % Use x or xw 

% display result
W = reshape(opttheta, params.numFeatures, params.n);
W=l2rowscaled(W,1);
figure,
imagesc(W)
end

% L1 L1 : i=-0 à 4 step 1
% L1 L2 : i=-2 à -10 step 2

%display_network(W');
% Dans un premier temps, L1L1 se comporte comme L1L2(ou l'inverse?) ->
% copie des filtres
% Puis il y a une spécification (les filtres choississent des supports
% finis) et enfin les ifltres s'orthogonalisent
% A un moment donné le trade off fait qu'il faut des filtres à suport fini



