clear all;
close all;
clc;

Screen_0=[550,10,800,650];
Screen_1=[1400,10,800,650];
Screen_2=[1360,-310,1920,1024];
Position=Screen_0;
Plots=1;
%% ---------------------------------------------------------
% Parameters
% ----------------------------------------------------------
d1=0.026438:.001:100;
d2=0.000001:.001:.05;
c=2.99*10^8;
f=900*10^6;
lamda=c/f;
B=[1 1 1 1 1];
theta=[0 0 0 0 0];
W=1;
T=1/(W);
freq=0:.001:W;
H=zeros(1,length(f));
%% ---------------------------------------------------------
% Free space path loss model for large distances
% ----------------------------------------------------------
L1=20*log10(((4*pi.*d1*f)/c));
figure('position',Position)
subplot(1,2,1)
plot(d1,-L1)
grid on;
axis([-10 100 -80 10]);
title({['Free Space Path Loss'];...
['as a Function of Distance']},'FontSize',16);
xlabel('Distance [m]');
ylabel('Magnitude [dB]');

%% ---------------------------------------------------------
% Free space path loss model for small distances
% ----------------------------------------------------------
L2=20*log10(((4*pi.*d2*f)/c));
subplot(1,2,2)
plot(d2,-L2)
grid on;
axis([-.005 0.05 -10 100]);
title({['Free Space Path Loss Model'];...
['as a Function of Distance'];...
['For Small Distances.']},'FontSize',16);
xlabel('Distance [m]','FontSize',12);
ylabel('Magnitude [dB]','FontSize',12);


%% ---------------------------------------------------------
% Doppler shift as a function of antenna speed and angle
% relative to the transmit antenna
% ----------------------------------------------------------
f_c=900*10^6;
c=2.99*10^8;
v=[0:2.25:100];
angle=linspace(-pi,pi,length(v));
[X,Y]=meshgrid(v,angle);
Z=X/c*f_c.*cos(Y);

figure('position',Position)
surf(X,Y,Z)%,'EdgeColor','none')
colormap(jet);
hold on
x=0:.01:100;
y=-pi/2*ones(1,length(x));
z=zeros(1,length(x));
plot3(x,y,z,'-.k','linewidth',2)
plot3(x,-y,z,'-.k','linewidth',2)
plot3(zeros(1,length(x)),y,linspace(-400,0,length(x)),...
'-.k','linewidth',2)
plot3(zeros(1,length(x)),-y,linspace(-400,0,length(x)),...
'-.k','linewidth',2);
title('Doppler Shift as a Function of Speed and Angle ... Relative to Trasmit Antenna')

xlabel('Speed [m/s]','FontSize',12);
ylabel('Angle [Radians]','FontSize',12);
zlabel('Doppler Shift [Hz]','FontSize',12)
set(gca,'YTick',-pi:pi/2:pi)
set(gca,'YTickLabel',{'-pi','-pi/2','0','pi/2','pi'})

