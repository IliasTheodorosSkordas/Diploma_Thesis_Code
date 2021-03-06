function varargout = Alamouti(varargin)

if nargin == 0  % LAUNCH GUI

	fig = openfig(mfilename,'reuse');

	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
	guidata(fig, handles);

	if nargout > 0
		varargout{1} = fig;
	end

elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK

	try
		if (nargout)
			[varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
		else
			feval(varargin{:}); % FEVAL switchyard
		end
	catch
		disp(lasterr);
	end

end

% --------------------------------------------------------------------
function varargout = Start_Callback(h, eventdata, handles, varargin)

%Get sytem parameters from GUI
N=str2double(get(handles.N_input,'String')); %total number of symbol pairs to be transmitted (should be at least 10 times more than expected 1/min(BER))
M=str2double(get(handles.M_input,'String')); %PSK order (must be a power of 2): 2, 4, 8 etc'
Tx=str2double(get(handles.Tx_input,'String')); %number of Tx elements, must be 2
Rx=str2double(get(handles.Rx_input,'String')); %number of Rx elements
SNR=eval(get(handles.SNR_input,'String')); %SNR in dB, average received power at one Rx element over the average noise power at that element

%Simulation parameters for the verson without GUI
%N=1000; %total number of symbol pairs to be transmitted (should be at least 10 times more than expected 1/min(BER))
%M=2;   %PSK order (must be a power of 2): 2, 4, 8 etc'
%SNR=0:10; %SNR in dB, average received power at one Rx element over the average noise power at that element
%Tx=2; %number of Tx elements, must be 2
%Rx=1; %number of Rx elements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Resets the generators to their initial state
randn('state',0); %Remove it if you want a random start of the randn generator
rand('state', 0); %Remove it if you want a random start of the rand generator

%Monte-Carlo
for k=1:length(SNR)
    %Toss pairs of uniformly distributed MPSK symbols with power 1/2
    A=floor(M*rand(2,N)); %transmitted alphabet
    st=exp(j*2*pi/M*A)/sqrt(2); %transmitted symbols
    
    %Simulate equivalent matrix of impulse noise
    %Noise power caculation
    snr=10^(SNR(k)/10); %just translate SNR from dB to times
    
    sig1=0.5/snr; %the sigma square of the noise
    
    Ns=sqrt(sig1)*(randn(2*Rx,N)+j*randn(2*Rx,N)); %noise matrix 
    
    %Transceiver
    for n=1:N
        %Toss the channel complex coefficients
        H=[]; %equivalent channel matrix initialization
        for r=1:Rx
            h=(randn(1,2)+j*randn(1,2))/sqrt(2); %Rayleigh channel
            %h=ones(1,2); %flat channel

            %Equivalent channel matrix:
            %h(1) - from first Tx to current Rx; h(2) - from second Tx to current Rx
            H=[H; h(1) h(2); h(2)' -h(1)'];
        end %m
    
        sr(:,n)=H'*H*st(:,n)+H'*Ns(:,n); %received symbols
    end %n
    
    %ML detection
    ang=angle(sr); %received angles
    B=mod(round(ang/(2*pi/M)),M); %received alphabet
   
    %BER estimation (for the Gray code constellations)
    BER(:,k)=sum(sum(xor(A-B,0)))/2/N/log2(M);
    
end %k (SNR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create plot
axes(handles.fig_ber)
semilogy(SNR, BER, 'k-v');
xlabel('SNR, [dB]');
ylabel('BER');
set(handles.fig_ber,'XMinorTick','on')
grid on


% --------------------------------------------------------------------
function varargout = N_input_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = M_input_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = SNR_input_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = Rx_input_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = Tx_input_Callback(h, eventdata, handles, varargin)





            
