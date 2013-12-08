A=rand(5,5)*2;
A=A/norm(A)*2;
A(abs(A)<0.1)=1;
B=rand(5,5)*2;
B=B/norm(B)*2;
B(abs(B)<0.1)=1;
C=rand(5,5)*2;
C=C/norm(C)*2;
C(abs(C)<0.1)=1;
epsilo=1e-5;

for i=0:8
    
    step=10^-i
   diff(i+1)=log(abs((costL1L2(A+step*B,C,epsilo)-costL1L2(A,C,epsilo))/step-trace(B'*gradientCostL1L2(A,C,epsilo))));
   
(costL1L2(A+step*B,C,epsilo)-costL1L2(A,C,epsilo))/step;
end

plot(diff)