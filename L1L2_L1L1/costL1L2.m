function cost=costL1L2(W,X,epsilon)

h=W*X;
tmp=sqrt(epsilon + h.^2);

cost=norm(sum(tmp,2))^2;



end