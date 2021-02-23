%
% Data file fmodels.m for E4C Flight Control Experiment.
% Copyright: Cambridge University Engineering Department, October 1994.
% Last modified: 17-01-95 by mcs.
%

%
% F4E Model 1.
% Altitude=5000ft. Mach 0.5.
%
a_f4e_1=[0 0 -1 0;0 -0.9896 17.41 96.15;0 0.2648 -0.8512 -11.39;0 0 0 -14];
b_f4e_1=[0 -97.78 0 14]';
c_f4e_1=[1 0 0 0];
d_f4e_1=0;
[num1,den1]=ss2tf(a_f4e_1,b_f4e_1,c_f4e_1,d_f4e_1,1);

%
% F4E Model 2.
% Altitude=5000ft. Mach 0.85.
%
a_f4e_2=[0 0 -1 0;0 -1.702 50.72 263.5;0 0.2201 -1.418 -31.99;0 0 0 -14];
b_f4e_2=[0 -272.2 0 14]';
c_f4e_2=[1 0 0 0];
d_f4e_2=0;
[num2,den2]=ss2tf(a_f4e_2,b_f4e_2,c_f4e_2,d_f4e_2,1);

%
% F4E Model 3.
% Altitude=35000ft. Mach 0.9.
%
a_f4e_3=[0 0 -1 0;0 -0.667 18.11 84.34;0 0.08201 -0.6587 -10.81;0 0 0 -14];
b_f4e_3=[0 -85.09 0 14]';
c_f4e_3=[1 0 0 0];
d_f4e_3=0;
[num3,den3]=ss2tf(a_f4e_3,b_f4e_3,c_f4e_3,d_f4e_3,1);

%
% F4E Model 4.
% Altitude=35000ft. Mach 1.5.
%
a_f4e_4=[0 0 -1 0;0 -0.5162 26.96 178.9;0 -0.6896 -1.225 -30.38;0 0 0 -14];
b_f4e_4=[0 -175.6 0 14]';
c_f4e_4=[1 0 0 0];
d_f4e_4=0;
[num4,den4]=ss2tf(a_f4e_4,b_f4e_4,c_f4e_4,d_f4e_4,1);

