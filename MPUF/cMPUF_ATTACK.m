clear all;
clc;

chalSize = 64;    % Bit length of challenge
mu = 0;           % Mean of variation in delay parametersd
sigma = 1;     % Standard deviation of variation in delay parameters
nXOR = 4;
sigmaNoise = sigma*0.10;
flaga=0; %active cma-es 1-used
flags=0; %sep cma-es 1-used
flagp=0; %spearman-1 or pearson-0
Sw= XORPUFgeneration(nXOR,chalSize,mu,sigma);
Dw= XORPUFgeneration(2^(nXOR-1),chalSize,mu,sigma);
nTrS =40000;
nTeS =5000;
zong=nTrS+nTeS;

Size = chalSize+1;
Evaluations =11;
U=50;
matchingRateMultipleTimes = zeros(U,nXOR);
matchingRateMultipleTimes1 = zeros(U,2^nXOR);

zongS= randi([0 1], zong, chalSize);
Phi_zongS = Transform(zongS, zong, chalSize);
[select1,AResponse_zongS] = ComputeNoisyResponsescMPUF(Dw,Sw,nXOR,Phi_zongS,zong,Size,chalSize,sigma,sigmaNoise,Evaluations);
%the example data
% f1="dataset/4_64_cMPUF_Phi_45000_0.10.csv";
% f2="dataset/4_64_cMPUF_Response_45000_0.10.csv";
% f3="dataset/4_64_cMPUF_Sw_45000_0.10.csv";
% Phi_zongS = csvread(f1);
% AResponse_zongS = csvread(f2);
% Sw=csvread(f3);
Phi_TrS = Phi_zongS(1:nTrS,:);
AResponse_TrS=AResponse_zongS(1:nTrS,:);
Phi_TeS = Phi_zongS(nTrS+1:zong,:);
AResponse_TeS=AResponse_zongS(nTrS+1:zong,:);

AR_TeS=zeros(nTeS,1);
for i=1:nTeS
    sum=0;
    for j=1:Evaluations
        sum=sum+AResponse_TeS(i,j);
    end
    if sum>5.5
        AR_TeS(i)=1;
    else
        AR_TeS(i)=0;
    end
end

modelS=zeros(nXOR,Size);
modelD=zeros(2^(nXOR-1),Size);

num=zeros(2^(nXOR-1),1);
CD=zeros(nXOR,nTrS,Size);
RD=zeros(nXOR,nTrS,1);
AResponseSAPUFs = zeros(nXOR,nTeS);
AResponseDAPUFs = zeros(2^(nXOR-1),nTeS);
for i=1:nXOR 
    AResponseSAPUFs(i,:)=ComputeResponseXOR(Sw(i,:),1,Phi_TeS,nTeS,Size);
end

InformReliability = ComputeTheNoiseInformation(AResponse_TrS,nTrS,Evaluations);
flag=0;
if flagp==1
    InformReliability=tiedrank(InformReliability);
end
for k=1:U
    opts=[];
    wModel = cmaes('modelAcc',rand(Size,1),0.5,opts,Phi_TrS,InformReliability,flaga,flags,flagp);

    for i=1:nXOR
       matchingRateMultipleTimes(k,i)=modelAccHa(wModel,Phi_TeS,AResponseSAPUFs(i,:),Size,nTeS);
       if((matchingRateMultipleTimes(k,i)>0.9)||(matchingRateMultipleTimes(k,i)<0.1))
           modelS(i,:)=wModel;
       end
    end
    for i=1:nXOR
        if modelS(i,1)==0
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

if flag==1
    [select,neg]=ComputeselectcMPUF(modelS,nXOR,Phi_TrS,nTrS,Size);
    for i=1:nTrS
        temp=select(i);
        num(temp)=num(temp)+1;
        temp1=num(temp);
        CD(temp,temp1,:)=Phi_TrS(i,:);
        RD(temp,temp1,:)=AResponse_TrS(i,1); 
        if neg(i)==1
            RD(temp,temp1,:)=~RD(temp,temp1,:);
        end
    end
    for i=1:2^(nXOR-1)
        temp=num(i);
        tCD=CD(i,1:temp,:);
        tCD=reshape(tCD,temp,Size);
        tRD=RD(i,1:temp);
        tRD=tRD';
        split=floor(temp*0.9);
        [allac, precision, recall, fscore,W]=LR_XAPUF(tCD(1:split,:),tRD(1:split,:),tCD(split+1:end,:),tRD(split+1:end,:),chalSize,1);
        modelD(i,:)=W;
    end
    [AResponse_TeS1,select] = ComputeResponsecMPUF(modelD,modelS,nXOR,Phi_TeS,nTeS,Size);
    acc=0;
    for i=1:nTeS
        if AResponse_TeS1(i)==AR_TeS(i)
            acc=acc+1;
        end
    end
    acc=acc/nTeS;
end



