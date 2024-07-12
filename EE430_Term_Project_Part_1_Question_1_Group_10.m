%% EE430 
%% Term Project
%%% Group 10
% Kerem Yakutlu - 2444107
% Pelin Ormanci - 2443620
%% Part 1
% *Question 1*

[y, fs] = audioread("SoundSampleWindows.wav");
sound(y, fs); %Playing the sound.
disp(fs + " is the sampling frequency.");
%% 
% Due to Nyquist-Shannon Sampling Theorem and the fact that human ear can hear 
% up to 20 kHz, we have to use a sampling frequency more than or equal to 40 kHz. 
% 
% From the outcome, the sampling rate of the WAV file is 24 kHz.

t = (1:length(y))/fs; %Getting the time passed.
plot(t,y); %Plotting the time domain signal.
hold on;
plot(0.1,0,'*r'); %W
plot(0.2,0,'*r'); %I
plot(0.4,0,'*r'); %N
plot(0.6,0,'*r'); %D
plot(0.7,0,'*r'); %O
plot(0.8,0,'*r'); %W
plot(1.1,0,'*r'); %S
hold off;

WindowSize = 6;
Overlap = 2;
NFFT = 10;

[s_Calculated, w_Calculated, t_Calculated] = calculateSpectrogram(y, fs, WindowSize, Overlap, NFFT); %Using our code
%[s_MATLAB, w_MATLAB, t_MATLAB] = spectrogram(y, hamming(Window), Overlap, NFFT, fs);
%Output_MATLAB_Spectrogram = spectrogram(y, hamming(Window), Overlap, NFFT,
%fs); For testing.

% Plotting the spectrogram.
figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');
colorbar;
%% 
% In this question, one may use Hamming window with window size of 6 and overlap 
% of 2. 
% 
% For the STFT, the window function should have a time-domain length about 100 
% ms - 200 ms. 
% 
% ...
%% 
% *Functions*

function mySpectrogram(signal, fs, windowSize, overlap, nfft)
    % Input:
    %   signal: Input signal
    %   fs: Sampling frequency of the signal
    %   windowSize: Size of the analysis window
    %   overlap: Overlap between consecutive windows (in samples)
    %   nfft: Number of FFT points

    % Calculate the spectrogram
    [S, F, T] = calculateSpectrogram(signal, fs, windowSize, overlap, nfft);

    % Plot the spectrogram
    figure;
    imagesc(T, F, 10*log10(abs(S)));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram');
    colorbar;
end

%% 
% 
% 
% 
% 
% 

function [S, F, T] = calculateSpectrogram(signal, fs, windowSize, overlap, nfft)
    % Calculate the spectrogram manually

    % Calculate the number of overlapping samples
    overlapSamples = overlap;

    % Calculate the number of windows
    numWindows = fix((length(signal) - overlapSamples) / (windowSize - overlapSamples));

    % Initialize the spectrogram matrix
    S = zeros(nfft/2 + 1, numWindows);

    % Perform the STFT
    for i = 1:numWindows
        % Extract the current window
        startIdx = (i-1)*(windowSize - overlapSamples) + 1;
        endIdx = startIdx + windowSize - 1;
        x = signal(startIdx:endIdx);

        % Apply the window function
        x = x .* hamming(windowSize);

        % Compute the FFT
        X = fft(x, nfft);

        % Store the magnitude spectrum in the spectrogram matrix
        S(:, i) = abs(X(1:nfft/2 + 1));

        % Store the corresponding time and frequency vectors
        T(i) = startIdx / fs;
        F = linspace(0, fs/2, nfft/2 + 1);
    end
end