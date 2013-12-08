function G=gradientsqrtBBt(B,T)

[U,S,V]=svd(B);



n=size(B,1);


M=S*V'*T'*U+U'*T*V*S';

for i=1:n
    for j=1:n
    weightD(i,j)=S(i,i)*S(j,j)*(S(i,i)+S(j,j));
    end
end
N=-M./weightD;
G=U*N*U';

end