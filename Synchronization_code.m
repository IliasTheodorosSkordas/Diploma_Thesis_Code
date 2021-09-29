%% -- Frequency Offset -------------------------------------
%
% This Matlab script will simulate a frequency offset in
% OFDM.
% ----------------------------------------------------------


% -- Scipt Parameters --------------------------------------
%
% Screen 0 =Size and position of plots for screen 0
% Screen 1 =Size and position of plots for screen 1
% Screen 1 =Size and position of plots for dual monitor
% Position =Plot screen variable
% Plots =Toggle to turn plots on(1) or off(0)
% ----------------------------------------------------------
Screen_0=[550,10,800,650];

Position=Screen_0;
% Plots=1;

%% -- Parameters -------------------------------------------
%
%
% ----------------------------------------------------------
frames=120;
Nfft=64;
bins=52;
cp=16;
M=16;
num=log2(M);
x4=[];

Epsilon=0.05;

SNR=20;
Theta=0.0;
x3save=[];
x4save=[];

%% -- OFDM Time Series -------------------------------------
%
% x1 =Random 16-QAM data
% x2 =Input to IFFT
% x3 =Output of IFFT
% x4 =Time series output of modulator
% ----------------------------------------------------------
for nn=1:frames
    x1=(floor(num*rand(1,bins+1))-(num-1)/2)/((num-1)/2)+...
        1j*(floor(num*rand(1,bins+1))-(num-1)/2)/((num-1)/2);
    x1(27)=0;
    x2=zeros(1,Nfft);
    x2((-26:26)+Nfft/2+1)=x1;
    x2=fftshift(x2);
    x3=4*ifft(x2);
    x3save(nn,:)=[x3(end-cp+1:end) x3];
    x4(nn,:)=x3save(nn,:).*exp(1j*2*pi*((nn-1)*(cp+Nfft)...
        :nn*(cp+Nfft)-1)/Nfft*Epsilon).*exp(1j*Theta);
    x5(nn,:)=x3save(nn,:).*exp(1j*2*pi*(0:Nfft+cp-1)/...
        Nfft*Epsilon).*exp(1j*Theta);
end
x4=awgn(x4,SNR,'measured');
x5=awgn(x5,SNR,'measured');

%% -- Demodulate With Offset -------------------------------
%
% y1 =Time series with frequency offset and noise
% reg1 =Input to FFT
% reg2 =Output of FFT
% reg3 =Used frequency bins
% reg4 =Demodulated symbols
% ----------------------------------------------------------
for nn=1:frames
    reg1=x4(nn,cp+1:cp+Nfft);
    reg1a=x5(nn,cp+1:cp+Nfft);
    reg2=fftshift(fft(reg1))/4;
    reg2a=fftshift(fft(reg1a))/4;
    reg3=[reg2((-26:-1)+Nfft/2+1) reg2((1:26)+Nfft/2+1)];
    reg3a=[reg2a((-26:-1)+Nfft/2+1) reg2a((1:26)+Nfft/2+1)];
    reg4(nn,:)=reg2;
    reg4a(nn,:)=reg2a;
    reg5(nn,:)=reg3;
    reg5a(nn,:)=reg3a;
end

figure('name','Frequency Offset Simulation- ... Constellation Diagram With Offset', 'position', Position)
plot(reg5a,'ro')
grid on 
title({['Constellation of ',num2str(M),...
    '-QAM with Normalized CFO of \epsilon=',...
    num2str(Epsilon)];['for ',num2str(frames),...
    ' Simulations of Transmitting One Frame']},...
    'FontSize',16)
axis([-1.5 1.5 -1.5 1.5])
axis('square')
hold on
plot(sqrt(2)*1/3*exp(1j*2*pi*(0:0.01:10)),'k')
plot(sqrt(10)/3*exp(1j*2*pi*(0:0.01:10)),'k')
plot(sqrt(2)*exp(1j*2*pi*(0:0.01:10)),'k')

figure('name',' Frequency Offset Simulation-...  Constellation Diagram With Offset','position',Position)
plot(reg5,'ro')
grid on
title({['Constellation of ',num2str(M),...
    '-QAM with Normalized CFO of \epsilon=',...
    num2str(Epsilon)]},'FontSize',16)
axis([-1.5 1.5 -1.5 1.5])
axis('square')
hold on
plot(sqrt(2)*1/3*exp(1j*2*pi*(0:0.01:10)),'k')
plot(sqrt(10)/3*exp(1j*2*pi*(0:0.01:10)),'k')
plot(sqrt(2)*exp(1j*2*pi*(0:0.01:10)),'k')


figure('name',' Frequency Offset Simulation-... Spectrum With Offset','position',Position)

subplot(2,1,1)
plot(-32:31,real(reg4),'rx')
grid on
xlabel('Bin','Fontsize',12)
title('Real Part','FontSize',12)

subplot(2,1,2)
plot(-32:31,imag(reg4),'bx')
grid on
xlabel('Bin','Fontsize',12)
title('Imaginary Part','FontSize',12)

[ax,h3]=suplabel('Spectrum of 16-QAM with Normalized CFO...\epsilon=0.05','t');
set(h3,'FontSize',16)

%% -- ICI Investigation ------------------------------------
x=0:63;
ICI1=exp(1j*pi*(x+Epsilon)*(1+1/Nfft))/Nfft.*...
    ((sin(pi*(x+Epsilon)))./(sin(pi*(x+Epsilon)/Nfft)));
ICI2=exp(1j*pi*(x+Epsilon/2)*(1+1/Nfft))/Nfft.*...
    ((sin(pi*(x+Epsilon/2)))./(sin(pi*(x+Epsilon/2)/Nfft)));
ICI3=exp(1j*pi*(x+Epsilon*2)*(1+1/Nfft))/Nfft.*...
    ((sin(pi*(x+Epsilon*2)))./(sin(pi*(x+Epsilon*2)/Nfft)));
ICI4=exp(1j*pi*(x+Epsilon*10)*(1+1/Nfft))/Nfft.*...
    ((sin(pi*(x+Epsilon*10)))./(sin(pi*(x+Epsilon*10)/Nfft)));

figure('name',' Frequency Offset Simulation- ICI ...152 Coefficients','position',Position)
plot(x,20*log10(abs(ICI4)),'gp')
hold on
plot(x,20*log10(abs(ICI3)),'k*')
plot(x,20*log10(abs(ICI1)),'ro')
plot(x,20*log10(abs(ICI2)),'x')
axis([0 64 -60 5])
grid on
title('Interchannel Interference Coefficients',...
    'FontSize',16)
xlabel('Subcarrier Index','FontSize',12)
ylabel('Power [dB]','FontSize',12)
legend({['\epsilon=',num2str(Epsilon*10)];...
    ['\epsilon=',num2str(Epsilon*2)];...
    ['\epsilon=',num2str(Epsilon)];...
    ['\epsilon=',num2str(Epsilon/2)]},...
    'location','Best','FontSize',12)

Epsilon=0:0.01:0.5;
x=1:63;
ICI_Power=[];
for nn=Epsilon
    ICI_Power=[ICI_Power;sum(abs(exp(1j*pi*(x+nn)*...
        (1+1/Nfft))/Nfft.*((sin(pi*(x+nn)))./...
        (sin(pi*(x+nn)/Nfft)))).^2)];
end
figure('name',' Frequency Offset Simulation-...ICI Coefficients','position',Position)
plot(Epsilon,10*log10(ICI_Power))
hold on
plot(Epsilon,10*log10(((pi*Epsilon).^2)/3),'-xr')
axis([0 0.5 -35 0])
grid on
xlabel('Normalized Carrier Frequency Offset','FontSize',12);
ylabel('Power [dB]','FontSize',12);
title('ICI Power Resulting From Frequency Offset',...
    'FontSize',16);
legend('Theoretical','Approximation','location','northwest')

k=1:Nfft-1;
den=[];
num=[];
t=0.00001:0.001:0.5;
for nn=t
    num=[num abs(exp(1j*pi*(nn)*(1+1/Nfft))/Nfft.*...
        ((sin(pi*(nn)))./(sin(pi*(nn)/Nfft)))).^2];
    den=[den sum(abs((exp(1j*pi*(k+nn)*(1+1/Nfft))/Nfft)...
        .*(sin(pi*(k+nn))./sin(pi*(k+nn)/Nfft))).^2)];
end

figure('name',' Frequency Offset Simulation- ...Carrier to Interference Power Ratio (CIR)','position',Position)
plot(t,10*log10(num./den))
xlabel('Normalized Carrier Frequency Offset \epsilon',...
    'FontSize',12)
ylabel('CIR [dB]','FontSize',12)
title('Carrier to Interference Power Ratio (CIR)',...
    'FontSize',16)
grid on
axis([-0.01 0.5 -5 100])
hold on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -- SampleClock.m ----------------------------------------

% This MATLAB simulation shows the effects of sampling clock
% offset.
% ----------------------------------------------------------

% -- Scipt Parameters --------------------------------------
%
% Screen 0 =Size and position of plots for screen 0
% Screen 1 =Size and position of plots for screen 1
% Screen 1 =Size and position of plots for dual monitor
% Position =Plot screen variable
% Plots =Toggle to turn plots on(1) or off(0)
% ----------------------------------------------------------
Screen_0=[550,10,800,650];

Position=Screen_0;
Plots=1;

%% -- Parameters -------------------------------------------
Nsym=64;
symSpace=1;
Nfft=64;
numSims=100;
fs=100;
epsilon=0.01;
sampOffset=round((1+epsilon)*fs);
iciPower=zeros(1,Nfft);
sigPower=zeros(1,Nfft);
M=4;
num=log2(M);
k=0:Nfft-1;
atten=1/Nfft*((1-exp(1j*2*pi*k*epsilon))./...175
    (1-exp(1j*2*pi*k*epsilon/Nfft)));

atten(1)=1;
figure('name',' Sample Clock Offset Simulation-...Spectrum','position',Position)
hold on
for t=1:numSims
    x1=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
        +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
    
    
    % Demodulate
    y=ifft(x1);
    y1=fs*ifft(x1,Nfft*fs);
    
    y1=y1(1:sampOffset:end);
    y1=[y1 zeros(1,Nfft-length(y1))];
    Y=fft(y,Nfft);
    Y1=fft(y1,Nfft);
    
    ici=Y1-x1.*atten;
    iciPower=iciPower+abs(ici/sqrt(2)).^2;
    sigPower=sigPower+abs(x1.*atten).^2;
    
    
    
    subplot(3,1,1)
    plot(0:Nfft-1,real(Y),'rx')
    grid on
    hold on
    subplot(3,1,2)
    plot(0:Nfft-1,real(Y.*atten),'rx')
    hold on
    grid on
    subplot(3,1,3)
    plot(0:Nfft-1,real(Y1),'rx')
    hold on
    grid on
end

%% -- ICI Investigation ------------------------------------
figure(1)
subplot(3,1,1)
title('No Sample Clock Offset','FontSize',12)
axis([0 64 -1.5 1.5])

subplot(3,1,2)
title({['Sample Clock Offset of ',num2str(epsilon*100),...
    '% With No ICI']},'FontSize',12)
axis([0 64 -1.5 1.5])

subplot(3,1,3)
xlabel('Sub-carrier Index','FontSize',12)
title({['Sample Clock Offset of ',num2str(epsilon*100),...
    '% With ICI']},'FontSize',12)
axis([0 64 -3 3])
[ax,h3]=suplabel('Effects of Sample Clock Offset on Spectrum','t');
set(h3,'FontSize',16)


figure('name',' Sampling Clock Offset Simulation- ...Constellation Diagram','position',Position)
plot(1:64,real(y(1:64)),'-o')
hold on
plot(1:64,real(y1(1:64)),'-*r')
grid on
legend('No Offset','Resampled With Offset')
xlabel('Sample','FontSize',12)
ylabel('Magnitude','FontSize',12)
title('Time Series with 1% Sample Clock Offset','FontSize',16)

figure('name',' Sampling Clock Offset Simulation- ...Constellation Diagram','position',Position)
subplot(2,2,1)
plot(Y,'bo')
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
title('Constellation With No Offset','FontSize',12)
grid on
axis('square')
axis([-2 2 -2 2])

subplot(2,2,2)
plot(Y.*atten,'rx')
hold on
plot(Y,'bo')
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
title('Constellation With Offset No ICI','FontSize',12)
grid on
axis('square')
axis([-2 2 -2 2])

subplot(2,1,2)
plot(Y1,'rx')
hold on
plot(Y,'bo')
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
title('Constellation With Offset and ICI','FontSize',12)
grid on
grid on
axis('square')
axis([-2 2 -2 2])

[ax,h3]=suplabel('Constellation Diagram For Sampling...Clock Offset of 1%','t');
set(h3,'FontSize',16')


figure('name',' Sample Clock Offset Simulation-...Magnitude of Attenuation and Phase of kˆ{th} ...    Sub-carrier Index','position',Position)
subplot(2,1,1)
plot(0:Nfft-1,10*log10(abs(atten)))
grid on
title('Magnitude of Attenuation on kˆ{th} Sub-carrier',...
    'FontSize',12)
xlabel('Sub-carrier Index','FontSize',12)
ylabel('Magnitude [dB]','FontSize',12)
axis([0 64 -4 0])

subplot(2,1,2)
plot(0:Nfft-1,angle(atten))
grid on
title('Phase Offset on kˆ{th} Sub-carrier','FontSize',12)
xlabel('Sub-carrier Index','FontSize',12)
ylabel('Phase [Rad]','FontSize',12)
axis([0 64 0 2.2])

[ax,h3]=suplabel('Attenuation and Phase Offset From 1%...Sampling Offset on kˆ{th} Sub-carrier','t');
set(h3,'FontSize',16)

for nn=0:Nfft-1
    ICI=sin(pi*((1+epsilon)*k-nn))./sin(pi/Nfft*...
        ((1+epsilon)*k-nn));
    ICI_power(nn+1)=sum(abs(ICI((1:Nfft)~=(nn+1))).^2)...
        /Nfft^2;
end

figure('name',' Sample Clock Offset Simulation- ...ICI Power','position',Position)
plot(k,10*log10(ICI_power),'r')
hold on
plot(k,10*log10(iciPower/numSims),'bo')
grid on
xlabel('Sub-carrier Index','FontSize',12)
ylabel('Power [dB]','FontSize',12)
title('ICI Power','FontSize',16)
legend({['Theoretical'];['Simulated']},...
    'location','northwest','FontSize',12)
axis([0 64 -20 0])

figure('name',' Sample Clock Offset Simulation- ...Signal to ICI Power Ratio','position',Position)
plot(k,10*log10(sigPower./ICI_power))
grid on
xlabel('Sub-carrier Index','FontSize',12)
ylabel('Power [dB]','FontSize',12)
title('Signal to ICI Power Ratio','FontSize',16)
axis([0 64 15 40])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -- SymbolTimingOffset.m ---------------------------------

% This MATLAB simulation shows the effects of frame or
% symbol timing offset.
% ----------------------------------------------------------


% -- Scipt Parameters --------------------------------------
%
% Screen 0 =Size and position of plots for screen 0
% Screen 1 =Size and position of plots for screen 1
% Screen 1 =Size and position of plots for dual monitor
% Position =Plot screen variable
% Plots =Toggle to turn plots on(1) or off(0)
% ----------------------------------------------------------
Screen_0=[550,10,800,650];
Screen_1=[1400,10,800,650];
Screen_2=[];
Position=Screen_0;
Plots=1;

%% -- Parameters -------------------------------------------
numSims=60;
bins=52;
zeta=8;
ICI_Power=0;
ISI_Power=0;
Interference=0;
M=4;
num=sqrt(M);

%% -- Simulate symbol offset with no cyclic prefix ---------
for Nfft=[64 256]
    atten=(Nfft-zeta)/Nfft*exp(1j*2*pi*zeta*(0:Nfft-1)/Nfft);
    y1=[];
    ICI_Power=0;
    ISI_Power=0;
    for nn=1:numSims
        x0_i=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
            +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
        x0_ii=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
            +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
        
        x1_i=ifft(x0_i);
        x1_ii=ifft(x0_ii);
        
        % -- Compute the demodulated symbol with attenuation
        X1=x0_i.*atten;
        
        % -- Demodulate offset frame
        y1(nn,:)=fft([x1_i(zeta+1:end) x1_ii(1:zeta)]);
        
        % -- Compute the ICI by demodulating frame 1 and
        % subtracting the attenuation
        ICI=fft([x1_i(zeta+1:end) zeros(1,zeta)])-atten;
        
        % -- Compute the ISI by demodulating frame 2 only
        ISI=fft([zeros(1,Nfft-zeta) x1_ii(1:zeta)]);
        
        
        
        ICI_Power=ICI_Power+sum(abs(ICI).^2);
        ISI_Power=ISI_Power+sum(abs(ISI).^2);
        
        Interference=Interference+sum(abs(y1(nn,:)-X1).^2);
    end
    
    % -- Compute average power of ICI,ISI, and Inteference
    Avg_ISI_power=ISI_Power/numSims;
    Avg_ICI_power=ICI_Power/numSims;
    Avg_Interference_power=Interference/numSims;
    
    % -- Compute ICI,ISI, and interference power from equations
    ICI_theory=(Nfft-zeta)*zeta/Nfft^2;
    ISI_theory=zeta/Nfft;
    Interference_theory=(2*Nfft-zeta)*zeta/Nfft^2;
    
    figure('name',' Frame Timing Offset Simulation-...    Effects of Frame Timing Offset','position',Position)
    plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
    hold on
    plot(y1(numSims,:),'rx')
    axis([-2 2 -2 2])
    axis('square')
    grid on
    xlabel('In-Phase','FontSize',12)
    ylabel('Quadrature','FontSize',12)
    title({['Symbols with Frame Offset with N {fft}=',...
        num2str(Nfft)]},'FontSize',16)
    
    figure('name',' Frame Timing Offset Simulation-...    Effects of Frame Timing Offset','position',Position)
    subplot(2,2,1)
    plot(x0_i,'bo')
    hold on
    plot(X1,'rx')
    axis([-2 2 -2 2])
    axis('square')
    grid on
    xlabel('In-Phase','FontSize',12)
    ylabel('Quadrature','FontSize',12)
    title('Attenuation and Rotation')
    
    subplot(2,2,2)
    plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
    hold on
    plot(ICI,'rx')
    axis([-3 3 -3 3])
    axis('square')
    grid on
    xlabel('In-Phase','FontSize',12)
    ylabel('Quadrature','FontSize',12)
    title('ICI contribution')
    
    subplot(2,2,3)
    plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
    hold on
    plot(ISI,'rx')
    axis([-1.2 1.2 -1.2 1.2])
    axis('square')
    grid on
    xlabel('In-Phase','FontSize',12)
    ylabel('Quadrature','FontSize',12)
    title('ISI Contributon')
    
    subplot(2,2,4)
    plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
    hold on
    plot(y1(numSims,:),'rx')
    axis([-2 2 -2 2])
    axis('square')
    grid on
    xlabel('In-Phase','FontSize',12)
    ylabel('Quadrature','FontSize',12)
    title('Symbols with Frame Offset')
    
    if Nfft==64
        [ax,h3]=suplabel('Effects of Frame Offset \zeta=8 Nfft=64','t');
        set(h3,'FontSize',16)
    else
        [ax,h3]=suplabel('Effects of Frame Offset \zeta=8 Nfft=256','t');
        set(h3,'FontSize',16)
    end
    
    figure('name',' Frame Timing Offset Simulation-...    Effects of Frame Timing Offset','position',Position)
    subplot(2,1,1)
    plot(-Nfft/2:Nfft/2-1,real(y1),'bx')
    grid on
    title('Real Part of Spectrum','FontSize',12)
    xlabel('Sample','FontSize',12)
    
    subplot(2,1,2)
    plot(-Nfft/2:Nfft/2-1,imag(y1),'rx')
    grid on
    title('Imaginary Part of Spectrum','FontSize',12)
    xlabel('Sample','FontSize',12)
    
    if Nfft==64
        [ax,h3]=suplabel('Effects of Frame Offset \zeta=8 Nfft=64','t');
        set(h3,'FontSize',16)
    else
        [ax,h3]=suplabel('Effects of Frame Offset \zeta=8 Nfft=256','t');
        set(h3,'FontSize',16)
    end
end

%% -- Simulate symbol offset with cyclic prefix ------------
Nfft=64;
cp=Nfft/4;
for nn=1:numSims
    x0_i=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
        +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
    x0_ii=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
        +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
    
    x1_i=ifft(x0_i);
    x1_ii=ifft(x0_ii);
    
    x2_i=[x1_i(end-cp+1:end) x1_i];
    x2_ii=[x1_ii(end-cp+1:end) x1_ii];
    
    reg1=[x2_i(end-zeta+1:end) x2_ii(1:end-zeta)];
    reg2=reg1(cp+1:end);
    y2(nn,:)=fft(reg2);
end

figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
'position',Position)
plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
hold on
plot(y2,'rx')
hold off
grid on
legend('No Offset','Offset')
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
title('Frame Timing Offset \zeta=8 With Cyclic Prefix',...
    'FontSize',16)
axis('square')



%% -- Simulation with channel and cyclic prefix ------------
h1=[1 0 0.1j 0.1 0 0 0.05j];
h2=[1 0 0 0.3 0 0.25j 0 0.1+0.15j];
Nfft=64;
cp=16;

figure
subplot(2,1,1)
plot(1:Nfft,20*log10(abs(fftshift(fft(h1,Nfft)))))

subplot(2,1,2)
plot(1:Nfft,abs(ifft(fft(h1,Nfft))))
for nn=1:numSims
    x0_i=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
        +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
    x0_ii=(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2)...
        +1j*(floor(num*rand(1,Nfft))-(num-1)/2)/((num-1)/2);
    
    x1_i=ifft(fftshift(x0_i));
    x1_ii=ifft(fftshift(x0_ii));
    
    x2_i=[x1_i(end-cp+1:end) x1_i];
    x2_ii=[x1_ii(end-cp+1:end) x1_ii];
    
    x3_i=filter(h1,1,x2_i);
    x3_ii=filter(h1,1,x2_ii);
    
    reg1=[x3_i(end-zeta+1:end) x3_ii(1:end-zeta)];
    reg2=[x2_i(end-zeta+1:end) x2_ii(1:end-zeta)];
    reg3=reg1(cp+1:end);
    reg4=reg2(cp+1:end);
    
    y3(nn,:)=fftshift(fft(reg3));
    y4(nn,:)=fftshift(fft(reg4));
end

figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
    'position',Position)
plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
hold on
plot(y4,'rx')
hold off
grid on
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
legend('No Offset','Offset')
title('Frame Timing Offset \zeta=8 With Cyclic Prefix ...Ideal Channel','FontSize',16)
axis('square')


figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
    'position',Position)
subplot(2,1,1)
plot(-Nfft/2:Nfft/2-1,real(y4),'bx')
grid on
title('Real Part of Spectrum','FontSize',12)
xlabel('Samples','FontSize',12)

subplot(2,1,2)
plot(-Nfft/2:Nfft/2-1,imag(y4),'rx')
grid on
title('Imaginary Part of Spectrum','FontSize',12)
xlabel('Samples','FontSize',12)

[ax,h3]=suplabel('Effects of Symbol Timing Offset With ...Cyclic Prefix With No Channel','t');
set(h3,'FontSize',16)


figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
    'position',Position)
plot(x0_i,'bo','MarkerSize',6,'LineWidth',2)
hold on
plot(y3,'rx')
hold off
grid on
xlabel('In-Phase','FontSize',12)
ylabel('Quadrature','FontSize',12)
legend('No Offset','Offset')
title('Frame Timing Offset \zeta=8 With Cyclic Prefix ...Mutipath channel','FontSize',16)
axis('square')



figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
    'position',Position)
subplot(2,1,1)
plot(-Nfft/2:Nfft/2-1,real(y3),'bx')
grid on
title('Real Part of Spectrum','FontSize',12)
xlabel('Sample','FontSize',12)

subplot(2,1,2)
plot(-Nfft/2:Nfft/2-1,imag(y3),'rx')
grid on
title('Imaginary Part of Spectrum','FontSize',12)
xlabel('Sample','FontSize',12)

[ax,h3]=suplabel('Effects of Symbol Timing Offset With...Cyclic Prefix and Channel','t');
set(h3,'FontSize',16)


%% -- Interference Power -----------------------------------
Nfft=64;
zeta=0:Nfft/2;

ICIpower=(Nfft-zeta).*zeta/Nfft^2;
ISIpower=zeta/Nfft;
Total=(2*Nfft-zeta).*zeta/Nfft^2;

figure('name',' Frame Timing Offset Simulation-...Effects of Frame Timing Offset With Cyclic Prefix',...
    'position',Position)
subplot(2,2,1)
plot(zeta,10*log10(ICIpower))
grid on
title('Power in ICI','FontSize',12)
xlabel('Offset [\zeta]','FontSize',12)
ylabel('Power [dB]','FontSize',12)

subplot(2,2,2)
plot(zeta,10*log10(ISIpower))
grid on
title('Power in ISI','FontSize',12)
xlabel('Offset [\zeta]','FontSize',12)
ylabel('Power [dB]','FontSize',12)

subplot(2,1,2)
plot(zeta,10*log10(Total))
grid on
title('Total Interference','FontSize',12)
xlabel('Offset [\zeta]','FontSize',12)
ylabel('Power [dB]','FontSize',12)

[ax,h3]=suplabel('Interference Power for N {fft}=64','t');
set(h3,'FontSize',16)