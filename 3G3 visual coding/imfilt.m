function out=imfilt(x,y)
nimage=size(x,2);
sz=sqrt(size(x,1));
out=zeros(sz*sz,1);
bsz=sqrt(numel(y));
y=reshape(y,bsz,bsz);

w=reshape(x,sz,sz);
out=reshape(filter2(y,w),sz*sz,1);

m1=min(min(out));
m2=max(max(out));

out=out/m2;


%out=(out-m1)/(m2-m1);