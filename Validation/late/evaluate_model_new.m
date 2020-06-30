clc
clear all

load('expre.mat');
load('reg_ability_pvalue.mat');

AIC_value=[];
expre_new=csvread('stage3.gc.csv', 1, 1); % GSE26899

for i=1:size(reg_ability_pvalue,1)
Y=[];Y=expre(i,:)';
extra_num=1;reg_id=find(reg_ability_pvalue(i,1:(end-1))~=0);
theta=[];theta=[reg_ability_pvalue(i,reg_id)';reg_ability_pvalue(i,end)];
phi=[];phi=expre(reg_id,:);
phi=cat(1,phi,ones(1,length(Y))); phi=phi';
resn=[];resn=sum((Y-phi*theta).^2);
sigma_2=resn/length(Y);
AIC_value(i,1)=log(sigma_2)+2*length(theta)/length(Y);
end

for k=1:1000
for i=1:size(reg_ability_pvalue,1)
Y=[];Y=expre(i,randperm(size(expre,2)))';
extra_num=1;reg_id=find(reg_ability_pvalue(i,1:(end-1))~=0);
theta=[];theta=[reg_ability_pvalue(i,reg_id)';reg_ability_pvalue(i,end)];
phi=[];
for j=1:length(reg_id)
phi=[phi;expre(reg_id(j),randperm(size(expre,2)))];
end
phi=cat(1,phi,ones(1,length(Y))); phi=phi';
resn=[];resn=sum((Y-phi*theta).^2);
sigma_2=resn/length(Y);
AIC_value(i,k+1)=log(sigma_2)+2*length(theta)/length(Y);
end
end

for i=1:size(reg_ability_pvalue,1)
Y=[];Y=expre_new(i,:)';
extra_num=1;reg_id=find(reg_ability_pvalue(i,1:(end-1))~=0);
theta=[];theta=[reg_ability_pvalue(i,reg_id)';reg_ability_pvalue(i,end)];
phi=[];phi=expre_new(reg_id,:);
phi=cat(1,phi,ones(1,length(Y))); phi=phi';
resn=[];resn=sum((Y-phi*theta).^2);
sigma_2=resn/length(Y);
AIC_value(i,1000+2)=log(sigma_2)+2*length(theta)/length(Y);
end

ind=[];
for i=1:size(AIC_value,1)
    if sum(AIC_value(i,end)>AIC_value(i,2:(end-1)))==0
        ind=[ind;i];
    end
end

Genesignificant=length(ind)/size(AIC_value,1);% models p-value<0.001
idinf=find(isinf(AIC_value(:,end)));
AIC_value1=AIC_value;
AIC_value1(idinf,:)=[]; % new data
reg_ability_pvalue1=reg_ability_pvalue;reg_ability_pvalue1(idinf,:)=[];
ConNum=[];for i=1:size(reg_ability_pvalue1,1);ConNum(i,1)=length(find(reg_ability_pvalue1(i,1:(end-1))~=0));end
pv=[];for i=1:size(AIC_value1,1);pv(i,1)=(length(find(AIC_value1(i,end)>AIC_value1(i,2:(end-1))))+1)/1000;end
pv1=[];for i=1:size(AIC_value1,1);pv1(i,1)=(length(find(AIC_value1(i,1)>AIC_value1(i,2:(end-1))))+1)/1000;end
figure;subplot(1,2,1);plot(ConNum,-log10(pv),'xk');xlabel('Connection Number');ylabel('-log10(Pvalue)');title('New data Comp. to permutation');subplot(1,2,2);plot(ConNum,-log10(pv1),'xk');xlabel('Connection Number');ylabel('-log10(Pvalue)');title('Ori. data Comp. to permutation');
