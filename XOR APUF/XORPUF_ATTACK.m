clear all;
clc;

chalSize = 64;    % Bit length of challenge
mu = 0;           % Mean of variation in delay parametersd
sigma = 1;     % Standard deviation of variation in delay parameters  
nXOR = 4;
flaga=0; %active cma-es 1-used
flags=0; %sep cma-es 1-used
flagp=0; %spearman-1 or pearson-0

nTrS =15000;
nTeS =5000;

Size = chalSize+1;


Evaluations =11;
sigmaNoise = 0.10;
U=50;

matchingRateMultipleTimes = zeros(U,nXOR);
XORw= XORPUFgeneration(nXOR,chalSize,mu,sigma);

TeS= randi([0 1], nTeS, chalSize);
Phi_TeS = Transform(TeS, nTeS, chalSize);
AResponse_TeS1 = ComputeResponseXOR(XORw,nXOR,Phi_TeS,nTeS,Size);


flag=0;
TrS= randi([0 1], nTrS, chalSize);
Phi_TrS = Transform(TrS, nTrS, chalSize);
AResponse_TrS = ComputeNoisyResponsesXOR(XORw,nXOR,Phi_TrS,nTrS,Size,chalSize,sigma,sigmaNoise,Evaluations);
model=zeros(nXOR,Size);
InformReliability = ComputeTheNoiseInformation(AResponse_TrS,nTrS,Evaluations);
% use the example data
% f1="dataset/4_64_TrS_Phi_15k_0.10.csv";
% f2="dataset/4_64_Reliability_15k_0.10.csv";
% f3="dataset/4_64_TeS_Response_15k_0.10.csv";
% f4="dataset/4_64_TeS_Phi_15k_0.10.csv";
% f5="dataset/4_64_XORw_15k_0.10.csv";
% Phi_TrS=csvread(f1);
% InformReliability=csvread(f2)';
% AResponse_TeS1=csvread(f3)';
% Phi_TeS=csvread(f4);
% XORw=csvread(f5);
%to verify the found APUF
AResponseALLAPUFs = zeros(nXOR,nTeS);
for i=1:nXOR 
    AResponseALLAPUFs(i,:)=ComputeResponseXOR(XORw(i,:),1,Phi_TeS,nTeS,Size);
end
if flagp==1
    InformReliability=tiedrank(InformReliability);
end
%cma-es attack
for k=1:U
    opts=[];
    wModel = cmaes('modelAcc',rand(Size,1),0.5,opts,Phi_TrS,InformReliability,flaga,flags,flagp);
    for i=1:nXOR
       matchingRateMultipleTimes(k,i)=modelAccHa(wModel,Phi_TeS,AResponseALLAPUFs(i,:),Size,nTeS);
       if((matchingRateMultipleTimes(k,i)>0.9)||(matchingRateMultipleTimes(k,i)<0.1))
           model(i,:)=wModel;
       end
    end
    for i=1:nXOR
        if model(i,1)==0
            break;
        end
        if i==nXOR
            flag=1;
        end
    end
    if flag ==1
       break;
    end
    
end 
if(flag==1)
   acc=modelAcc(model,Phi_TeS,AResponse_TeS1,Size,nTeS,nXOR);
end