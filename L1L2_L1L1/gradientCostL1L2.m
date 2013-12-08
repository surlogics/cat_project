function g=gradientCostL1L2(W,X,epsilon)

h=W*X;

tmp=sqrt(epsilon + h.^2);

square2coeff=2*(sum(tmp,2));

g=(h./tmp);
g=g*X';

g=bsxfun(@times,g,square2coeff);

%cost=norm(sum(tmp,2));



end