function B=normalize_bases(B,A)

nbase=size(B,2);
VAR_GOAL=0.1;
persistent gain A_var;
if isempty(gain)
    gain=ones(nbase,1);
    A_var=VAR_GOAL*ones(nbase,1);
end

nswatch=size(A,2);
var_eta=0.001;
alpha=0.02;
VAR_GOAL=0.1;

for i=1:nswatch
    A_var = (1-var_eta)*A_var + var_eta*A(:,i).*A(:,i);
end
gain = gain .* ((A_var/VAR_GOAL).^alpha);
normA=sqrt(sum(B.*B));

for i=1:nbase
    B(:,i)=gain(i)*B(:,i)/normA(i);
end