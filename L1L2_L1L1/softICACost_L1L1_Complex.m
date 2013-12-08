function [cost,grad] = softICACost2(theta, x, params)
% L1L1 POOLING
% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);




X=x;

% Forward Prop
h = W*x;
r = W'*h;

% Sparsity Cost
%K = sqrt(params.epsilon + h.^2);
%sparsity_cost = params.lambda * sum(sum(K));
%K = 1./K;

% Reconstruction Loss and Back Prop
diff = (r - x);
reconstruction_cost = 0.5 * sum(sum(diff.^2));
outderv = diff;


% energy cost



Wcomplex=W(1:2:end,:)+1i*W(2:2:end,:);
y=Wcomplex*X;
%weights=2*sum(abs(Wcomplex*X));
absval=sqrt(params.epsilon+abs(y).^2);%real(Wcomplex*X).^2+imag(Wcomplex*X).^2);

%sum(sum(absval))-sum(sum(sqrt(params.epsilon + h.^2)))
SGcomplex=(y./absval)*X';%*diag(weights);
% ADD on
SG(1:2:2*params.n,:)=real(SGcomplex);
SG(2:2:2*params.n,:)=imag(SGcomplex);




en=params.lambda*sum(sum(absval));
%en=norm(sum(sqrt(tmp(1:2:end,:).^2+tmp(2:2:end,:).^2),2));

% compute the cost comprised of: 1) sparsity and 2) reconstruction
cost = en + reconstruction_cost;



% Backprop Output Layer
W2grad = outderv * h';

% Baclprop Hidden Layer
outderv = W * outderv;
W1grad = outderv * x';
Wgrad = params.lambda*SG + W1grad+ W2grad';



% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);