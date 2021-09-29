clc;
clear all;
close all;

% -- Plot Parameters ---------------------------------------
%
% Screen 0 =Size and position of plots for screen 0
% Screen 1 =Size and position of plots for dual monitors
% Screen 1 =Size and position of plots for HDMI Out
% (1080i)
% Position =Plot screen variable
% Plots =Toggle to turn plots on(1) or off(0)
% ----------------------------------------------------------
Screen_0=[550,10,800,650];
Screen_1=[1400,10,800,650];
Screen_2=[1360,-310,1920,1024];
Position=Screen_0;
Plots=1;

%% ---------------------------------------------------------
% Calculate superposition of waveforms for phase differences
% of {0, pi/8, pi/4, pi/2, 3pi/4, pi}.
% ----------------------------------------------------------
d=[0 pi/8 pi/4 pi/2 3*pi/4 pi];
for i=1:length(d)
if d(i)==0
a=('Full Wavelength');
elseif d(i)==pi/8
a=('1/16 Wavelength');
elseif d(i)==pi/4;
a=('1/8 Wavelength');
elseif d(i)==pi/2
a=('1/4 Wavelength');
elseif d(i)==3*pi/4
a=('3/8 Wavelength');
elseif d(i)==pi
a=('1/2 Wavelength');
else
a=('');
end

t=0:.01:10;
f=sin(2*pi*.5.*t);
g=sin(2*pi*.5.*t-d(i));
h=f+g;
figure('position',Position)
subplot(2,1,1)
plot(t,f,'LineWidth',3)
grid on
hold on
plot(t,g,'--r','LineWidth',2)
grid on
title({['Two Waveforms With Phase Difference ',a,'.']},...
'FontSize',16)
set(gca, 'XTick', [])
subplot(2,1,2)
plot(t,h,'k','LineWidth',2)
grid on
set(gca, 'XTick', [])
set(gca,'Ylim',[-2 2]);
title('The Received Waveform','FontSize',16)
end