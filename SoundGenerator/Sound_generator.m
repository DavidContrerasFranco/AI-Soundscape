close all;
clear;
clc;

% Data Loading
Fs = 16000;
Data = load('../Freq_components/freqcomponents.mat'); 
Matrix = Data.Answ; 

% Matrix Simplification
Freq_Components = Matrix(:,1,:) + Matrix(:,2,:).*j;
Freq_Components = squeeze(Freq_Components);

% Inverse fourier transform
Time_domain = ifft(Freq_Components,[],1);
Time_domain = Time_domain(:);

% Audio sample
Audio = abs(Time_domain);

% Generate an audio file
audiowrite('freqcomponents.wav',Audio,Fs);

