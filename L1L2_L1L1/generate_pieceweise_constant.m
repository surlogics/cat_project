function X=generate_pieceweise_constant(number,N,K)
% function X=generate_pieceweise_constant(number,N,K)
%index=ceil(rand(K-1,number)*N);
randvalues=rand(K,number);
%index(end+1,number)=N;
%tmp=index;
%index(1,:)=1;
%index(2:end+1,:)=tmp;


for i=1:number
    A=zeros(1,N);
    index=rand(1,K+1);
    index=cumsum(index);
    index=floor(N*index/max(index));
    
    index(1)=1;
    index(index==0)=1;
    for k=1:K
        A(index(k):index(k+1))=randvalues(k,i);
    end
    
    X(:,i)=A;


end


X=bsxfun(@minus,X,mean(X));
X=bsxfun(@times,X,1./std(X));

end