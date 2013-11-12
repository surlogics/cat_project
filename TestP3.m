function TestP3(iLam)

% Test P3

addpath(genpath('./rica/'));

load('mnist_norm.mat');
load('RandIndices.mat');

NClass = 10;
NPCTr = 500;
NPCTe = 20;

%ThVec = [(0.001:0.002:0.009)'; (0.1:0.1:1)'] * 0.04;
ThVec = [(0.001:0.002:0.009)'; (0.1:0.1:0.5)'] * 0.1;
%ThVec = 0;

numlam=20;      % number of lambda values
maxlam=1000;    % range of lambda values
fac=log10(maxlam);
lamVec=10.^(fac*[1:numlam]/numlam);

NRound = 5;

numOut = 100;

ilam = num2str(iLam);


acc=zeros(NRound,length(lamVec),length(ThVec));

% Normalizaiton
          
Xte = normc(Xte);
Xtr = normc(Xtr);

for iRound = 1:NRound

IdxTr = IdxTrSet{iRound};
IdxTe = IdxTeSet{iRound};

D = Xtr(:,IdxTr);
T = Xte(:,IdxTe);
TrainLabel = Ytr(IdxTr)+1;
TestLabel = Yte(IdxTe)+1;

%for ilam = 1:length(lamVec)

% start of dictionary learning (class specific)
lambda = lamVec(ilam);
Dp = [];
for i = 1:NClass
    [Di,cost]=LearnDict(D(:,TrainLabel == i), lambda, numOut);
    Dp = [Dp; Di];
end
D = Dp;
TrainLabel = reshape(repmat((1:NClass), numOut,1),NClass*numOut,1);
% end of dictionary learning

p = size(D, 2);
numt = size(T,2);

for iTh = 1:length(ThVec)
    
% Soft thresholding
Th = ThVec(iTh);%0.02;

% Prediction w/o scaling
R = zeros(numt,NClass);
for i = 1:NClass
    Di = D(TrainLabel == i, :);
    R(:,i) = sqrt(sum(((Di' * wthresh(Di * T,'s',Th))'- T').^2,2));
end
[minvalue PredictLabel] = min(R, [], 2);  
acc(iRound, ilam, iTh) = sum(PredictLabel == TestLabel)/length(TestLabel);


end


%end

end

% [accTh, maxidx] = max(mean(acc,1),[],1);
% 
% figure;
% plot(ThVec,accTh,'ro-');

% save('TestP3_result.mat','acc','lamVec','ThVec','maxidx','accTh');
save(sprintf('./TestOutput/TestP3_result_ilam%d.mat',ilam),'acc','lamVec','ThVec');

% ----------------------------------------------------------------------------%
% Result:
% 
% Soft thresholding + class specific dictionary learning doesn't make much 
% difference to the CRC-RLS and sub-RLS.
%
% ----------------------------------------------------------------------------%


