%
% Command file flysim.m for 3F1 Flight Control Experiment.
% Copyright: Cambridge University Engineering Department, October 1994.
% Author: M.C. Smith.
%

num=2;den=[1 1];	%  Numerator and denominator of plant 
			%  Laplace transfer function

runtime=15;   	% target simulation interval in seconds

wght=[0,0,0,0];	% entries are: impulse, step and sinusoid disturbance
		% weightings and sinusoidal frequency (Hz). Impulse and step
		% occur randomly between 0.2 and 0.6 secs. Sinusoid 
	 	% begins at t=0.

samper=30;	% target sampling period in milliseconds

srate=(samper+1.3)/1000;	% anticipated average sampling period in secs
                            % was samper+0.6

grphc1

for i=1:count
	set(hh,'Xdata',hx,'Ydata',hy+y*hz);
	pp=p(1,2);
	pp=sign(pp)*min(max(0,abs(pp)-0.05),10);
	set(jh,'Xdata',jx,'Ydata',jy+pp*jz);
	drawnow;

	ylist(i)=y;
	ulist(i)=pp;

	x=adis*x + bdis*(pp+disturb(i));
	y=cdis*x + ddis*(pp+disturb(i));
	
	while (time2-time1<samper)
		time2=clock;time2=1000*(60*time2(5)+time2(6));
	end
	thetimes(i)=time2;
	time1=time2;
	
	if (y<-10 | y>10 )
		flg=1;crashind=i+1;
		thetimes(i+1)=thetimes(i)+samper;
		ylist(i+1)=y;
		ulist(i+1)=sign(p(1,2))*min(abs(p(1,2)),10);
		break;
	end

end

grphc2