close all;
clear;
clc;

fstsmp = 6; %First sample
time = 1; %Sample time
Fs = 44100; %Freq of the audio
pos = 0; %Index
Fsnew = 8000; %New Freq of the sample
nmbsmp = 599; %Number of samples
Answ = zeros(Fsnew,2,nmbsmp); %Size of the answer

for k = fstsmp:fstsmp+nmbsmp %Interval
    %If we are in the first sample.
    if k == 1
       samples = [1,time*Fs];
    else
        samples = [(k-1)*time*Fs + 1,k*time*Fs];
    end
    [y,Fs] = audioread('../Data/Recording_1',samples); %Sample extraction
    x = fft(y); %Fast Fourier Transform of the sample
    Answ(:,:,pos+1) = [real(x(1:Fsnew)) imag(x(1:Fsnew))]; %Real part and Img Part of the fft
    pos = pos +1; %New index
end
save ../Freq_components/freqcomponents.mat Answ