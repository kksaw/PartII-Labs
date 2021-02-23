function bodedisp(fmin,fmax,fnum)
%BODEDISP  Graphics display routine for 3F1 Flight Control Experiment.
%   BODEDISP(fmin,fmax,fnum) plots the Bode diagram for a continous-time
%   plant G(s) = num/den in series with a gain Kgain and a pure time
%   delay of Dtime sec, discretized at a samping rate of srate sec.  
%   The variables num, den, srate, Kgain and Dtime must be in the MATLAB
%   workspace.  (num, den, srate typically come from flysim or a similar
%   script; Kgain and Dtime should be entered at the MATLAB command
%   prompt.)  The minimum frequency is fmin, the maximum is fmax, and the
%   number of points evaluated is fnum.
%
%   BODEDISP(fmin,fmax) is the same as BODEDISP(fmin,fmax,100).
%
%   BODEDISP is the same as BODEDISP(0.1,pi/srate,100).
% 
%   Copyright: Cambridge University Engineering Department, October 1994.
%   Last modified: 22-10-03 by DJA.

try
  % Attempt to read values of num, den and srate from the workspace.
  num=evalin('base','num');
  den=evalin('base','den');
  samp=evalin('base','srate');
catch
  % Warn the user to run a file such as flysim first.
  disp('num, den and srate are undefined.')
  disp('You need to run a program such as flysim before using bodedisp.')
  return
end

try
  % Attempt to read values of Dtime and Kgain from the workspace.
  Dtime=evalin('base','Dtime');
  Kgain=evalin('base','Kgain');
catch
  % Warn the user to enter these by hand.
  disp('Either Dtime, Kgain, or both are undefined.')
  disp('You need to enter these at the command line, e.g.')
  disp(' ')
  disp('    >> Dtime = (your delay time)')
  disp('    >> Kgain = (your gain)')
  disp(' ')
  return
end

try
  % Attempt to read value of userid from the workspace.
  userid=evalin('base','userid');
catch
  % Prompt user for a value, and update the workspace.
  userid = '';
  while isempty(userid)
	userid = input('Please enter your userid: ','s');
  end
  assignin('base','userid',userid)
end
  
switch nargin
 case 0
  fmin = 0.1;
  fmax = pi/samp;
  fnum = 100;
 case 2
  fnum = 100;
 case 3
  % All is OK
 otherwise
  error('usage: bodedisp or bodedisp(fmin,fmax,fnum)');
end
if fmax > pi/samp
  disp('Maximum frequency has been reduced to pi/srate.')
  fmax = pi/samp;
end
wlist = logspace(log10(fmin),log10(fmax),fnum);


% Display the time delay, proportonal gain, numerator and denominator.
Dtime
Kgain
num
den

% Convert the numerator and denominator to a continuous-time transfer
% function. 
[acon,bcon,ccon,dcon]=tf2ss(num,den);

% Convert it to a discrete-time transfer function.
[adis,bdis]=c2d(acon,bcon,samp);
cdis=ccon;ddis=dcon;

nlag=ceil(Dtime/samp);
denlag=zeros(1,nlag+1);
denlag(1)=1;
[a1,b1,c1,d1]=tf2ss(Kgain,denlag);
[a2,b2,c2,d2]=series(a1,b1,c1,d1,adis,bdis,cdis,ddis);
figure(3)
clf reset
cla reset
dbode(a2,b2,c2,d2,samp,1,wlist);
grid on
axes('position',[.12,.1,.79,.82])
text(-0.05,1.06,num2str(userid))
axis('off')
title(PlotTitle(Kgain,Dtime))

disp('If your plot is not smooth enough, you can improve it by plotting')
disp('more points.  See the handout for details.')
disp(' ')

if ~nargin 
  disp('To find gain and phase margins, you need to be able to see the') 
  disp('180-degree line clearly.  If you need to, you can change the')
  disp('frequency range by using a command of the form')
  disp(' ')
  disp('    >> bodedisp(fmin,fmax,fnum)')
  disp(' ')
  disp('as described in the laboratory handout.')
  disp(' ')
end
  
return

function str = PlotTitle(Kgain,Dtime)
str = ['Plant in series with gain of ',num2str(Kgain,2)];
str = [str,' and ',num2str(Dtime,2),' sec delay'];
return

  