function y= myfilter(A,f)
sz=size(f,1);
if rem(sz,2)==0
    sz1=sz/2-1;
    sz2=sz/2;
else
    sz1=(sz-1)/2;
    sz2=(sz-1)/2;
end
L=A(:,sz1:-1:1);
R=A(:,end:-1:end-sz2);
A=[L A R];
T=A(sz1:-1:1,:);
B=A(end:-1:end-sz2,:);
A=[T; A ;B];

for j=1:size(A,1)-sz
    for k=1:size(A,2)-sz;
        y(j,k)=sum(sum(A(j:j+sz-1,k:k+sz-1).*f));
    end
end;