function en=energyL1L2(W,x,params)
% 
% 
% r=W'*W*x;
% 
% diff = (r - x);
% reconstruction_cost = 0.5 * sum(sum(diff.^2));
% outderv = diff;
% 
% 
% Wcomplex=W(1:2:end,:)+1i*W(2:2:end,:);
% y=Wcomplex*x;
% %weights=2*sum(abs(Wcomplex*X));
% %absval=sqrt(params.epsilon+abs(y).^2);%real(Wcomplex*X).^2+imag(Wcomplex*X).^2);
% absval=sqrt(params.epsilon+abs(y).^2);%real(Wcomplex*X).^2+imag(Wcomplex*X).^2);
%en=params.lambda*norm(sum(absval))^2+reconstruction_cost;

Wcomplex=W(1:2:end,:)+1i*W(2:2:end,:);
y=Wcomplex*x;
absval=sqrt(params.epsilon+abs(y).^2);

en=norm(sum(absval,2))^2;

end