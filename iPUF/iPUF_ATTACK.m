clear all;
clc;


chalSize = 64;    % Bit length of challenge
mu = 0;           % Mean of variation in delay parameters
sigma = 1;        % Standard deviation of variation in delay parameters
x =2;        % x - number of APUFs in x-XOR PUF
y =2;        % y - number of APUFs in y-XOR PUF
feedback_a =33;   % feedback position to connect the output of x-XOR PUF and 
                  % the y-XOR PUF,  0<=feedback_a<=chalSize-1 
Size = chalSize+1;
flaga=0; %active cma-es 1-used
flags=0; %sep cma-es 1-used
flagp=0; %spearman-1 or pearson-0

%generate (x,y)-MXPUF
prediction_array = zeros(1,y);
flag = zeros(1,y);
wModellist=zeros(y,Size+1);
[x_XPw,y_XPw]=MXPUFgeneration(x,y,chalSize,mu,sigma);

%generate Test Set and Training Set. 
xunhuancount=1;
zongxunhuan=0;
nTrS = 20000; %size of test set
trainSetChallenges= randi([0 1], nTrS, chalSize);
trainSetResponses = ComputeResponseMXPUF( ...
                       x_XPw,y_XPw,x,y,feedback_a, ...
                       trainSetChallenges,nTrS,chalSize ...
                       );                  

%for firstmodel=1:x
  [flag,prediction_array,wModellist] = MXPUF_getfirstmodel(x_XPw,y_XPw,x,y,trainSetChallenges,flag,prediction_array,wModellist,flaga,flags,flagp);
%end
[testchallenge,testresponse]=gettest_set(x_XPw,chalSize,x);
preac=0;
preWup=zeros(x,chalSize+1);
while xunhuancount<4&&zongxunhuan<24

zongxunhuan=zongxunhuan+1;

if(preac>0.8)
    xunhuancount=xunhuancount+1;
end

TrSp = zeros(nTrS,chalSize+1);
    for i=1:nTrS
        for j=1:(feedback_a-1)
            TrSp(i,j)= trainSetChallenges(i,j);
        end
        Interposbit= 0;
        TrSp(i,feedback_a)= Interposbit;
        for j=(feedback_a+1):(chalSize+1)
            TrSp(i,j) = trainSetChallenges(i,j-1);
        end                
    end
    Phi_TrSp = Transform(TrSp, nTrS, chalSize+1);
    
   Areponse0=ComputeResponseXOR(wModellist,y,Phi_TrSp,nTrS,size(Phi_TrSp,2));
   
TrSp = zeros(nTrS,chalSize+1);
    for i=1:nTrS
        for j=1:(feedback_a-1)
            TrSp(i,j)= trainSetChallenges(i,j);
        end
        Interposbit= 1;
        TrSp(i,feedback_a)= Interposbit;
        for j=(feedback_a+1):(chalSize+1)
            TrSp(i,j) = trainSetChallenges(i,j-1);
        end                
    end
    Phi_TrSp = Transform(TrSp, nTrS, chalSize+1);
    Areponse1=ComputeResponseXOR(wModellist,y,Phi_TrSp,nTrS,size(Phi_TrSp,2));
    LRtrainsetchallenges=[];
    LRtrainsetresponse=[];
    count=1;
    for i=1:nTrS
        if(Areponse0(i)~= Areponse1(i))
            if(Areponse0(i)==trainSetResponses(i))
                LRtrainsetchallenges(count,:)=trainSetChallenges(i,:);
                LRtrainsetresponse=[LRtrainsetresponse,0];
            else
                LRtrainsetchallenges(count,:)=trainSetChallenges(i,:);
                LRtrainsetresponse=[LRtrainsetresponse,1];
            end
            count=count+1;
        end
    end
LRtrainsetresponse=transpose(LRtrainsetresponse);
[Wup,ac1]= get_model_up(LRtrainsetchallenges,LRtrainsetresponse,x,testchallenge,testresponse,preac,preWup);

preac=ac1;
preWup=Wup;
[sum0,flag,prediction_array,wModellist] = get_model_down(x_XPw,y_XPw,x,y,Wup,trainSetChallenges,prediction_array,wModellist,flag,flaga,flags,flagp);

end

nTest = 5000; %size of test set
TestSetChallenges= randi([0 1], nTest, chalSize);
TestSetResponses = ComputeResponseMXPUF( ...
                       x_XPw,y_XPw,x,y,feedback_a, ...
                       TestSetChallenges,nTest,chalSize ...
                       ); 

TestY = ComputeResponseMXPUF2( ...
                       Wup,wModellist,x,y,feedback_a, ...
                       TestSetChallenges,nTest,chalSize ...
                       ); 
                   
[ac, precision, recall, fscore] = accuracy(TestSetResponses,TestY);



