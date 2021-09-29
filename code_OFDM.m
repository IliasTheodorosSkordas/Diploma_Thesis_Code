
 clear all;
 close all;
 clc;

 Screen 0=[550,10,800,650];
 Screen 1=[1400,10,800,650];
 Screen 2=[1360,-310,1920,1024];
 Position=Screen 0;
 Plots=1;

 %% -- Parameters -------------------------------------------
 %
 % N =Transform length
 % x =Input to IFFT (in serial to parallel form)
 % g =Number of bin outputs to graph
 % cp =Number of samples in cyclic prefix
 % ----------------------------------------------------------
 N=2ˆ6;
 g=9;
 cp=floor(0.25*N);
 x=diag(ones(1,N));
 %-- OFDM Modulation with IFFT ----------------------------
 Y=ifft(x);
 clear x;

 %% -- Plot IFFT Output for various bins --------------------
 figure('name',...
 'Real Part of IFFT Output',...
 'position',Position)
 for nn=1:g
 subplot(sqrt(g),sqrt(g),nn);
 plot(0:1/N:1-1/N,N*real(Y(:,nn)),'r');
 title({[num2str(nn-1),' Cycles per Interval']});
 axis([0 1 -1.1 1.1]);
 axis('square');
 set(gca,'YTick',[-1 0 1]);
 end
 figure(1)
 [ax4,h3]=suplabel('Real Part of 64-Point IFFT Output For...
 Frequency Bins 0-8','t');
 set(h3,'FontSize',16);

 figure('name',...
 'Imaginary Part of IFFT Output',...
 'position',Position)
 for nn=1:g
 subplot(sqrt(g),sqrt(g),nn);
 plot(0:1/N:1-1/N,N*imag(Y(:,nn)));
 title({[num2str(nn-1),' Cycles per Interval']});
 axis([0 1 -1.1 1.1]);
 axis('square');
 set(gca,'YTick',[-1 0 1]);
 end
 figure(2)
 [ax4,h3]=suplabel('Imaginary Part of 64-Point IFFT Output...
 For Frequency Bins 0-8','t');
 set(h3,'FontSize',16);

 figure('name',...
 'Real Part of IFFT Output',...
 'position',Position)

 stem(0:N-1,N*real(Y(:,4)),'r','LineWidth',2)
 hold on
 plot([0 0],[-.05 .05],'k','LineWidth',3);
 plot([63 63],[-.05 .05],'k','LineWidth',3);
 plot([-1 64],[0 0],'k','LineWidth',3);
 plot(0:N-1,N*real(Y(:,4)),'-.k','LineWidth',2)
 hold off
 axis([-1 N -1.1 1.1])
 set(gca,'YTick',[]);
 set(gca,'XTick',[])
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','0');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[100 355 15 20]);
 set(mTextBox,'BackgroundColor','w');
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','63');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[655 360 23 20]);
 set(mTextBox,'BackgroundColor','w');

 figure('name',...
 'Imaginary Part of IFFT Output'...
 ,'position',Position)
 stem(0:N-1,N*imag(Y(:,4)),'LineWidth',2)
 hold on
 plot([0 0],[-.05 0],'k','LineWidth',3);
 plot([63 63],[0 .05],'k','LineWidth',3);
 plot([-1 64],[0 0],'k','LineWidth',3);
 plot(0:N-1,N*imag(Y(:,4)),'-.k','LineWidth',2)
 hold off
 axis([-1 N -1.1 1.1])
 set(gca,'YTick',[]);
 set(gca,'XTick',[])
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','0');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[100 355 15 20]);
 set(mTextBox,'BackgroundColor','w');
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','63');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[655 360 23 20]);
 set(mTextBox,'BackgroundColor','w');
 %% -- OFDM Demodulation ------------------------------------
 x=fft(Y(:,4));
 figure('name',...
 'Real Part of FFT Output'...
 ,'position',Position)
 stem(real(x'),'r')
 set(gca,'XTick',[1 4 54 64])
set(gca,'XTickLabel',{'0','3','k','63'})
 axis([0 65 -0.05 2])
 set(gca,'cameraupvector',[-1 0 0]);
 set(gca,'YTick',[]);

 figure('name',...
 'Imaginary Part of FFT Output'...
 ,'position',[685,85,750,740])
 stem(imag(x'))
 set(gca,'XTick',[1 4 54 64])
 set(gca,'XTickLabel',{'0','3','k','63'})
 axis([0 65 -0.05 2])
 set(gca,'cameraupvector',[-1 0 0]);
set(gca,'YTick',[]);


 figure('name','Thesis OFDM Simulation- OFDM Spectrum'...
 ,'position',Position)
 f=-0.5:0.001:0.5-0.001;
 hold on
 y=0;
 for nn=0.05:0.05:0.25
 plot(f,sinc((f-nn)*20).*exp(1j*2*pi*(f)*.01))
 plot(f,sinc((f+nn)*20).*exp(1j*2*pi*(f)*.01))
 plot([nn nn],[0 1],'-.k')
 plot([-nn -nn],[0 1],'-.k')
 y=y+sinc((f-nn)*20)+sinc((f+nn)*20);
 end


 %% -- Cyclic Prefix ----------------------------------------
 Y cp=[Y(4,N-cp+1:N) Y(4,:)];
 figure('name','Cyclic Prefix'...
 ,'position',Position)

plot(0,0,'w')
 hold on
 axis([-0.1 79.1 -1.2 1.2])
 rectangle('Position',[N, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 rectangle('Position',[0, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 plot(0:N+cp-1,N*real(Y cp),'r')

 arrow([N 0],[cp-1 0],'Width',3)
 arrow([cp-1 1.1],[N+cp-1 1.1])
 arrow([N+cp-1 1.1],[cp-1 1.1])
 arrow([0 1.1],[cp-1.1 1.1])
 arrow([cp-1.1 1.1],[0 1.1])
 plot([cp-1 cp-1],[-1 1.2],'-.k')
 plot([N+cp-1 N+cp-1],[-1 1],'-.k')
 set(gca,'XTick',[])
 set(gca,'YTick',[-1 0 1])
 axis([-0.1 79.1 -1.2 1.2])
 title('Real Part of 25% Cyclically Extnded OFDM Symbol',...
 'FontSize',16)
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Prefix');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[125 650 60 20]);
 set(mTextBox,'BackgroundColor',[0.9 0.9 0.9]);
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Symbol');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[400 650 65 20]);
 set(mTextBox,'BackgroundColor','w');

 figure('name','Cyclic Prefix'...
 ,'position',Position)

 plot(0,0,'w')
 hold on
 axis([-0.1 79.1 -1.2 1.2])
 rectangle('Position',[N, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 rectangle('Position',[0, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 plot(0:N+cp-1,64*imag(Y cp))
 axis([-0.1 79.1 -1.2 1.2])
 arrow([N 0],[cp-1 0],'Width',3)
 arrow([cp-1 1.1],[N+cp-1 1.1])
 arrow([N+cp-1 1.1],[cp-1 1.1])
 arrow([0 1.1],[cp-1.1 1.1])
 arrow([cp-1.1 1.1],[0 1.1])
 hold on
 plot([cp-1 cp-1],[-1 1],'-.k')
 plot([N+cp-1 N+cp-1],[-1 1],'-.k')
 set(gca,'XTick',[])
 set(gca,'YTick',[-1 0 1])
 title(...
 'Imaginary Part of 25% Cyclically Extended OFDM Symbol'...
 ,'FontSize',16)
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Prefix');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[125 650 60 20]);
 set(mTextBox,'BackgroundColor',[0.9 0.9 0.9]);
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Symbol');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[400 650 65 20]);
 set(mTextBox,'BackgroundColor','w');



 figure('name','Cyclic Prefix'...
 ,'position',Position)

 plot(0,0,'w')
 hold on
 axis([-0.1 79.1 -1.2 1.2])
 rectangle('Position',[N, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 rectangle('Position',[0, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 stem(0:N+cp-1,N*real(Y cp),'r')

 arrow([N 0],[cp-1 0],'Width',3)
 arrow([cp-1 1.1],[N+cp-1 1.1])
 arrow([N+cp-1 1.1],[cp-1 1.1])
 arrow([0 1.1],[cp-1.1 1.1])
 arrow([cp-1.1 1.1],[0 1.1])
 plot([cp-1 cp-1],[-1 1.2],'-.k')
 plot([N+cp-1 N+cp-1],[-1 1],'-.k')
 set(gca,'XTick',[0 cp-1 N N+cp-1])
 set(gca,'YTick',[-1 0 1])
 axis([-0.1 79.1 -1.2 1.2])
 title('Real Part of 25% Cyclically Extnded OFDM Symbol',...
 'FontSize',16)
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Prefix');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[125 650 60 20]);
 set(mTextBox,'BackgroundColor',[0.9 0.9 0.9]);
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Symbol');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[400 650 65 20]);
 set(mTextBox,'BackgroundColor','w');

 figure('name','Cyclic Prefix'...
 ,'position',Position)
 plot(0,0,'w')
 hold on
 axis([-0.1 79.1 -1.2 1.2])
 rectangle('Position',[N, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 rectangle('Position',[0, -1.2, cp-1, 2.4],...
 'FaceColor',[0.9 0.9 0.9])
 stem(0:N+cp-1,64*imag(Y cp))
 axis([-0.1 79.1 -1.2 1.2])
 arrow([N 0],[cp-1 0],'Width',3)
 arrow([cp-1 1.1],[N+cp-1 1.1])
 arrow([N+cp-1 1.1],[cp-1 1.1])
 arrow([0 1.1],[cp-1.1 1.1])
 arrow([cp-1.1 1.1],[0 1.1])
 hold on
 plot([cp-1 cp-1],[-1 1],'-.k')
 plot([N+cp-1 N+cp-1],[-1 1],'-.k')
 set(gca,'XTick',[0 cp-1 N N+cp-1])
 set(gca,'YTick',[-1 0 1])
 title(...
 'Imaginary Part of 25% Cyclically Extended OFDM Symbol'...
 ,'FontSize',16)
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Prefix');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[125 650 60 20]);
 set(mTextBox,'BackgroundColor',[0.9 0.9 0.9]);
 mTextBox=uicontrol('style','text');
 set(mTextBox,'String','Symbol');
 set(mTextBox,'FontSize',14);
 set(mTextBox,'Position',[400 650 65 20]);
 set(mTextBox,'BackgroundColor','w');