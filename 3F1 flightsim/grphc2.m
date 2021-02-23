%
% Subroutine grphc2.m for 3F1 Flight Control Experiment.
% Copyright: Cambridge University Engineering Department, October 1994.
% Last modified: 11-08-03 by jpp.
%

thetimes=(thetimes -thetimes(1)*ones(count,1))/1000;
samptimes=diff(thetimes);

if flg==1,
thetimes=thetimes(1:crashind);
samptimes=samptimes(1:crashind-1);
ylist=ylist(1:crashind);
ulist=ulist(1:crashind);
end

minsamp=min(samptimes)*1000;
maxsamp=max(samptimes)*1000;
avgsamp=thetimes(length(thetimes))/(length(thetimes)-1)*1000;
samp=avgsamp/1000; %%%
disp(sprintf('The minimum, maximum and average sampling periods were: '))
disp(sprintf('%f , %f , %f milliseconds.',minsamp,maxsamp,avgsamp))
elist=-ylist;

figure(2);
clf reset
cla reset
plot(thetimes,elist,thetimes,ulist);
if ~exist('userid')
  userid=input('Type your userid: ','s');
end
axes('position',[.12,.1,.79,.82])
text(-0.05,1.06,num2str(userid))
axis('off')

if wght~=0 
	if distind <= crashind
	disp(sprintf('Disturbance occured at time %f seconds.',thetimes(distind)))
	end
end
if flg==1
disp(sprintf('You crashed at time %f seconds.',thetimes(crashind)))
end

