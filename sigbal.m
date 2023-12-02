% end to end communication model
clc
clear all
close all
%%signal generation
fm=1000; %modulating frequency
harm=[1 0.5 2 1];
amp=[1 2 3 1];
max_freq=max(fm*harm);
system_freq=20*max_freq;
system_sampling_rate=1/system_freq;
min_freq=min(fm*harm);
range=2/min_freq;
t=0:system_sampling_rate:range;

%generate the signal
msg=zeros(size(t));
for k=1:length(harm)
    msg=msg+amp(k)+sin(2*pi*harm(k)*fm*t);
end
figure(1);
plot(t,msg,"r");
grid on;
xlabel("time");
ylabel("amplitude");

%sampling 
n_sample=5;
fs=n_sample*max_freq;
msample=zeros(size(t));
b=1:system_freq/fs:length(t)
msample(1:system_freq/fs:length(t))=msg(1:system_freq/fs:length(t));
figure(2);
plot(t,msg,"r");
hold on
stem(t,msample)
legend("Generated signal", "sampled signal")
grid on
xlabel("time");
ylabel("amplitude");

%%quantization

no_of_levels=4;
quantile=(max(msample)-min(msample))/no_of_levels;
code=min(msample):quantile:max(msample)
mq=zeros(size(msample));
for k=1:length(code) %%For each quantization step
    values=msample>(code(k)-quantile/2) & msample<(code(k)+quantile/2)
    mq(values)=round(code(k)) %%For true values, it is lessen 
end
%%clear values
figure(3)
stem(t,mq,"r*");
grid on 
xlabel("time")
ylabel("Amplitude of the signal")

%%Encoding

mq1=mq-min(mq)
bits=de2bi(mq1(1:system_freq/(fs):length(mq)),4, "left-msb")
bits=bits(:); %% for linear alignment 

figure(4)
stem(bits,"*m");
hold on
legend("bits sequennce in the transmitter");

%%pass band modulation

fc=1e6; %1Mhz frequency
nsamp=10;
ncyn=2;
tb=0:1/(fc*nsamp):ncyn/fc; %1-bit is transmit in 2us as bit interval containing [0:0.1us:2us]-each bit interval can contain samples whose sample interval is 0.1us  
t_tran=0:1/(fc*nsamp): (ncyn*length(bits))/fc+(length(bits)-1)/(fc*nsamp); %size of the signal
mod_sig=zeros(size(t_tran));
l=1;
for k=1:length(tb):length(mod_sig)
    if (bits(l)==1)
        mod_sig(k:k+length(tb)-1)=cos(2*pi*fc*tb)
    else (bits(l)==1)
        mod_sig(k:k+length(tb)-1)=-cos(2*pi*fc*tb)
    end
    l=l+1
end

%%AWGN Channel
tran_signal=awgn(mod_sig,10);
figure(5)
plot(t_tran,mod_sig,".-b",t_tran,tran_signal,'r')
axis([0 3*ncyn/fc -2 2])
title("Transmitted and noies added recieved signal")
legend('transmittted ','recieved signal')
xlabel("time")
ylabel("amplitude")
grid on;

%% Recieved filter

f_freq=-(fc*nsamp)/2:(fc*nsamp)/length(t_tran):(fc*nsamp)/2 -(fc*nsamp)/length(t_tran)
f_tran=fft(tran_signal)
figure(7)
plot(f_freq,fftshift(f_tran),f_freq,fftshift(mod_sig))
xlabel("frequency")
ylabel("signal amplitude")
legend("modulated signal","recieved Signal") %%The signal is not filtered yet (f_tran=received signal mod_sig=transmitted signal), The recieved is low strength,,)
r_rece=zeros(size(f_tran))
fir=(f_freq<-3*fc | f_freq>3*fc)
f_rece(fir)=f_tran(fir);
f_rece(~fir)=0.5*f_tran(~fir);
t_rec=ifft(f_rece)

figure(8)

plot(t_tran,t_rec); %Filtered signal 
hold on
plot(t_tran,tran_signal); %After noise added
hold on
plot(t_tran,mod_sig); %Before noise added
% grid on;
xlabel("time");
ylabel("amplitude");
legend('Filtered signal in time domain','recieved signal in time domain','transmittted signal in time domin ')

% Demodultion & decoding  %% Correlate the prexsisting signal(+90-symbol 1) with the recieved signal
% if the correlation is greater than 0.5 then it is decoded recieved
% signal is 1 ,if it is not recieved symbol is 0

% Decoded bits are converted into equivalent integer voltages and using
% interpolation we could recover signal trans signal==recivered signal

dec_data=zeros(size(bits))
l=1;
for k=1:length(tb);length(t_tran)
a=corrcoef(cos(2*pi*fc*tb), t_rec(k:k+length(tb)-1))
b=mean(a);
    if (b>0.5) 
        dec_data(l)=1;
    else
        dec_data(l)=0;
    end
    l=l+1;
end
figure(9)
stem(dec_data,"b");
hold on
stem(bits,"*m");
grid on;
xlabel("bit position");
ylabel("bit sequence in the reciever vs transmitted");
legend("receieved", "transmitted")


% Decoding (converting binary to decimal)
dec_data=reshape(dec_data,4,length(dec_data)/4)';
mq_rece=zeros(size(mq));
mq_rece(1:system_freq/fs:length(mq))= bi2de(dec_data,"left-msb")'+min(mq);

%%signal reconstruction
f_freq=-1/(2*system_sampling_rate):1/(system_sampling_rate*length(t)):1/(2*system_sampling_rate)-1/(system_sampling_rate*length(t));
f_rece=fft(mq_rece);
f_out=zeros(size(f_rece));
figure(10);
plot(f_freq,fftshift(f_rece),f_freq,fftshift(fft(msg)));
grid on
xlabel("frequency");
ylabel("FTF msg and FFT recieved signal");
legend("Bit sequence in rec", "Bit sequence in Tx");

f_out(f_freq<-17000|f_freq>17000)=f_rece(f_freq<-17000|f_freq>17000);
gain=4;
out=ifft(f_out);
amplified_out=out*gain;
figure(11);
plot(t,amplified_out,t,msg);
grid on;
xlabel("time");
ylabel("recieved_signal vs transmitted_signal" );
legend("Rx signal","Trans_signal");















 




