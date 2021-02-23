%
% Subroutine grphc1.m for 3F1 Flight Control Experiment.
% Copyright: Cambridge University Engineering Department, October 1994.
% Last modified: 11-08-03 by jpp.
%

[acon,bcon,ccon,dcon]=tf2ss(num,den);
[adis,bdis]=c2d(acon,bcon,srate);
cdis=ccon;ddis=dcon;

format compact
figure(1);hold off;clf reset
drawnow
cla reset;axis([-11,11,-13,13]);axis('off')
set(1,'Color','k');
m10=line([5.8,7.2],[-10,-10],'LineStyle','-','Color','g');
m9=line([6.2,6.8],[-9,-9],'LineStyle','-','Color','g');
m8=line([6.2,6.8],[-8,-8],'LineStyle','-','Color','g');
m7=line([6.2,6.8],[-7,-7],'LineStyle','-','Color','g');
m6=line([6.2,6.8],[-6,-6],'LineStyle','-','Color','g');
m5=line([6.2,6.8],[-5,-5],'LineStyle','-','Color','g');
m4=line([6.2,6.8],[-4,-4],'LineStyle','-','Color','g');
m3=line([6.2,6.8],[-3,-3],'LineStyle','-','Color','g');
m2=line([6.2,6.8],[-2,-2],'LineStyle','-','Color','g');
m1=line([6.2,6.8],[-1,-1],'LineStyle','-','Color','g');
g0=line([5.8,7.2],[0,0],'LineStyle','-','Color','g');
g1=line([6.2,6.8],[1,1],'LineStyle','-','Color','g');
g2=line([6.2,6.8],[2,2],'LineStyle','-','Color','g');
g3=line([6.2,6.8],[3,3],'LineStyle','-','Color','g');
g4=line([6.2,6.8],[4,4],'LineStyle','-','Color','g');
g5=line([6.2,6.8],[5,5],'LineStyle','-','Color','g');
g6=line([6.2,6.8],[6,6],'LineStyle','-','Color','g');
g7=line([6.2,6.8],[7,7],'LineStyle','-','Color','g');
g8=line([6.2,6.8],[8,8],'LineStyle','-','Color','g');
g9=line([6.2,6.8],[9,9],'LineStyle','-','Color','g');
g10=line([5.8,7.2],[10,10],'LineStyle','-','Color','g');

hx=[-10,5];
hy=[0,0];
hz=ones(1,length(hx));
hh=line(hx,hy,'LineStyle','-','Color','r');

jx=[8.0,8.5,9.0,8.5,8.0];
jy=[0,0.25,0,-0.25,0];
jz=ones(1,length(jx));
jh=line(jx,jy,'LineStyle','-','Color','y');

set(gcf,'WindowButtonMotionfcn','p=get(gca,''CurrentPoint'');');
drawnow

%disp(sprintf('Press any key to begin.'))
%pause;disp(sprintf('Running.' ))
disp(sprintf('Get ready. Press any key to begin.'))
 pause 
;disp(sprintf('Running.' ))

count=floor(runtime*1000/(samper+1));

ylist=zeros(count,1);
ulist=zeros(count,1);
thetimes=zeros(count,1);
disturb=zeros(count,1);
distind=1+ceil((200+385*rand)/(samper+1));
disturb(distind)=-1000/(samper+1)*wght(1); 
disturb1=[zeros(distind-1,1);ones(count-distind+1,1)];
disturb=disturb-wght(2)*disturb1; 
tt=0:1:(count-1);
disturb1=wght(3)*sin(wght(4)*2*pi*srate*tt');
disturb=disturb-disturb1; 

crashind=count;

[n1,m1]=size(bdis);
x=zeros(n1,1);
y=0;
flg=0;
p=zeros(2,3);

time1=clock;time1=1000*(60*time1(5)+time1(6));
time2=time1;
start=time1;
