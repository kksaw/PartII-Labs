load images;
displayimages(I3,9);

%compact coding
[I, sz, nsamp] = [I1, 16, 1000];
[B1, percent1]=pca3g3(I,sz,nsamp);
displayimages(B1,16);
loglog(percent1);
overlayimages(I1,B1(:,1));

[I, sz, nsamp] = [I2, 16, 1000];
[B2, percent2]=pca3g3(I,sz,nsamp);
displayimages(B2,64);
loglog(percent2);
overlayimages(I2(:,1),B2(:,1));

%sparsecoding

s=randn(64,1); 
B=randn(64,64); 
a=randn(64,1); 
sigma=1; 
lambda=1; 
delta=0.1;

checkgrad(‘spfunc',a,delta,B,s,sigma,lambda);

load gradtest;
ainit = randn(64,1);
a = minimize(ainit,'spfunc',500,B,s,sigma,lambda);
plot(a,acorrect,'o');

tic; a = minimize(ainit,'spfunc',500,B,S,sigma,lambda);toc
tic; a = fast_minimize(B,s,sigma,lambda); toc

load gradtest;
dB=basefunc(acorrect,B,s);
plot(dB,dBcorrect,'o');


B=sparseopt(I3,500);



B2w = sparseopt(I2w,2000);
for k=1:64 %iterate over the bases
overlayimages(I2w(:,1), B(:,k), I2(:,1))
pause
end