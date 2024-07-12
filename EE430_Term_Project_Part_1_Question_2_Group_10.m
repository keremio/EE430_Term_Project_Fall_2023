%% EE430 
%% Term Project 
% *Question 2*
% Kerem Yakutlu - 2444107
% Pelin Ormanci - 2443620


% *Part i)*
% 
% In this question, we may look how to generate DTMF signals.

Symbols = {'1','2','3','4','5','6','7','8','9','*','0','#'};
lfg = [697 770 852 941]; % Low frequency group
hfg = [1209 1336 1477];  % High frequency group
f  = [];
for c=1:4
    for r=1:3
        f = [ f [lfg(c);hfg(r)] ];
    end
end
Fs  = 4000;       % Sampling frequency 8 kHz
N = 800;          % Tones of 100 ms
t   = (0:N-1)/Fs; % 800 samples at Fs
pit = 2*pi*t;

figure;
tones = zeros(N,size(f,2));
for toneChoice=1:12
    % Generate tone
    tones(:,toneChoice) = sum(sin(f(:,toneChoice)*pit))';
    % Plot tone
    subplot(4,3,toneChoice),plot(t*1e3,tones(:,toneChoice));
    title(['Symbol "', Symbols{toneChoice},'": [',num2str(f(1,toneChoice)),',',num2str(f(2,toneChoice)),']'])
    set(gca, 'Xlim', [0 25]);
    ylabel('Amplitude');
    if toneChoice>9, xlabel('Time (ms)'); end
end
hold off;
set(gcf, 'Color', [1 1 1], 'Position', [1 1 1280 1024])
annotation(gcf,'textbox', 'Position',[0.38 0.96 0.45 0.026],...
    'EdgeColor',[1 1 1],...
    'String', '\bf Time response of each tone of the telephone pad', ...
    'FitBoxToText','on');

%% 
% In order to create a DTMF signal in part i, for DTMF 4,

Fs = 4000;                                            % Sampling Frequency
t  = linspace(0, 1, Fs);                              % One Second Time Vector
w1 = 2*pi*770;                                        % Radian Value To Create DTMF 4 Tone
w2 = 2*pi*1209;                                       % Radian Value To Create DTMF 4 Tone
s_Pair_1 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_Pair_1, Fs)                                    % Produce Tone As Sound
%% 
% In order to create a DTMF signal in part i, for DTMF 3,

w1 = 2*pi*1477;                                       % Radian Value To Create DTMF 4 Tone
w2 = 2*pi*697;                                        % Radian Value To Create DTMF 3 Tone
s_DTMF3 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_DTMF3, Fs)                                    % Produce Tone As Sound
%% 
% In order to create a DTMF signal in part i, for DTMF 0,

w1 = 2*pi*1336;                                       % Radian Value To Create DTMF 4 Tone
w2 = 2*pi*941;                                        % Radian Value To Create DTMF 3 Tone
s_DTMF0 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_DTMF0, Fs)                                    % Produce Tone As Sound
%% 
% To create the combined sound signal, we may use a vector.

Combined_Signal = [s_Pair_1(1:40/1000*Fs), s_DTMF3(1:50/1000*Fs), s_DTMF0(1:60/1000*Fs)];
sound([s_Pair_1 s_DTMF3 s_DTMF0], Fs);
%% 
% Plotting the sound signal, in time domain.

t = (1:length(s_Pair_1))/Fs; %Getting the time passed.
figure;
plot(t(1:40/1000*Fs),s_Pair_1(1:40/1000*Fs)); %Plotting the time domain signal for DTMF 4
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("DTMF of 4");
figure;
plot(t(1:50/1000*Fs),s_DTMF3(1:50/1000*Fs)); %Plotting the time domain signal for DTMF 3
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("DTMF of 3");
figure;
plot(t(1:60/1000*Fs),s_DTMF0(1:60/1000*Fs)); %Plotting the time domain signal for DTMF 0
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("DTMF of 0");

t = (1:length(Combined_Signal))/Fs;
figure;
plot(t, Combined_Signal);
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("DTMF of Combined Sound Signal");
%% 
% Plotting the STFT of sound signal, by using Hamming window with a size of 
% 6.

WindowSize = 6;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTHamming(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Hamming Filter, Window Size = 6');
colorbar;

%% 
% Plotting the STFT of sound signal, by using Hamming window with a size of 
% 3.

WindowSize = 3;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTHamming(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Hamming Filter, Window Size = 3');
colorbar;

%% 
% Plotting the STFT of sound signal, by using Hamming window with a size of 
% 12.

WindowSize = 12;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTHamming(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Hamming Filter, Window Size = 12');
colorbar;

%% 
% Plotting the STFT of sound signal, by using rectangular window with a size 
% of 6.

WindowSize = 6;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTRectangular(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Rectangular Filter, Window Size = 6');
colorbar;

%% 
% Plotting the STFT of sound signal, by using rectangular window with a size 
% of 3.

WindowSize = 3;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTRectangular(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Rectangular Filter, Window Size = 3');
colorbar;

%% 
% Plotting the STFT of sound signal, by using rectangular window with a size 
% of 12.

WindowSize = 12;
Overlap = 2;
NFFT = 10;
[s_Calculated, w_Calculated, t_Calculated] = mySTFTRectangular(Combined_Signal, Fs, WindowSize, Overlap, NFFT); %Using our code

figure;
imagesc(t_Calculated, w_Calculated, 20*log10(abs(s_Calculated)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram with a Rectangular Filter, Window Size = 12');
colorbar;

%% 
% We may conclude that rectangular filter is cutting higher frequencies of the 
% signal, more than the Hamming filter.
% 
% Using smaller window size would be more precise attempt to calculate the STFT.
% 
% *Part ii)*
% 
% In order to create signals wanted in part ii, for the first pair,

Fs = 1000;                                            % Sampling Frequency
t  = linspace(0, 1e-1, Fs);                              % One Second Time Vector
w1 = 2*pi*100;                                        % Radian Value To Create Pair 1
w2 = 2*pi*110;                                        % Radian Value To Create Pair 1
s_Pair_1 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_Pair_1, Fs)                                    % Produce Tone As Sound
%% 
% In order to create signals wanted in part ii, for the second pair,

w1 = 2*pi*100;                                       % Radian Value To Create Pair 2
w2 = 2*pi*150;                                        % Radian Value To Create Pair 2
s_Pair_2 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_Pair_2, Fs)                                    % Produce Tone As Sound
%% 
% In order to create signals wanted in part ii, for the third pair,

w1 = 2*pi*100;                                       % Radian Value To Create Pair 3
w2 = 2*pi*200;                                        % Radian Value To Create Pair 3
s_Pair_3 = sin(w1*t) + sin(w2*t);                      % Create Tone
sound(s_Pair_3, Fs)                                    % Produce Tone As Sound
%% 
% Plotting the time-domain signals:

figure;
plot(t, s_Pair_1);
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("Pair 1");
figure;
plot(t, s_Pair_2);
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("Pair 2");
figure;
plot(t, s_Pair_3);
xlabel("Time (seconds)");
ylabel("Amplitude (V)");
title("Pair 3");

%% 
% Afterwards, calculating the spectrograms:

WindowSize = 6;
Overlap = 2;
NFFT = 10;
[S_Pair_1, F_Pair_1, T_Pair_1] = calculateSpectrogram(s_Pair_1, Fs, WindowSize, Overlap, NFFT);
[S_Pair_2, F_Pair_2, T_Pair_2] = calculateSpectrogram(s_Pair_2, Fs, WindowSize, Overlap, NFFT);
[S_Pair_3, F_Pair_3, T_Pair_3] = calculateSpectrogram(s_Pair_3, Fs, WindowSize, Overlap, NFFT);

%For pair 1
figure;
imagesc(T_Pair_1, F_Pair_1, 20*log10(abs(S_Pair_1)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram for Pair 1');
colorbar;

%For pair 2
figure;
imagesc(T_Pair_2, F_Pair_2, 20*log10(abs(S_Pair_2)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram for Pair 2');
colorbar;

%For pair 1
figure;
imagesc(T_Pair_3, F_Pair_3, 20*log10(abs(S_Pair_3)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram for Pair 3');
colorbar;

%% 
% To find the minimum number of frequencies, we may look at the fundamental 
% frequencies with respect to the pair frequencies and the sampling frequency.
% 
% For Pair 1,
% 
% $T_1 =\frac{100\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{1}{10}$ and $T_2 =\frac{110\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{11}{100}$
% 
% If the denominator is 100, we can define those periods with the denominator; 
% hence, that would be the minimum sampling points' number.
% 
% For Pair 2,
% 
% $T_1 =\frac{100\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{1}{10}$ and $T_2 =\frac{150\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{3}{20}$
% 
% The number is 20.
% 
% For Pair 3,
% 
% $T_1 =\frac{100\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{1}{10}$ and $T_2 =\frac{200\;\textrm{Hz}}{1\;k\textrm{Hz}}=\frac{1}{5}$
% 
% The number is 10.
% 
% *Part iii)*

DTMF_Ratio_Matrix = zeros(4,4);
%% 
% For DTMF 1, the ratio of frequencies is 

High_Frequency = 1209;
Low_Frequency = 697;
DTMF_Ratio_Matrix(1,1) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 1 frequencies.");
%% 
% For DTMF 2, the ratio of frequencies is 

High_Frequency = 1336;
Low_Frequency = 697;
DTMF_Ratio_Matrix(1,2) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 2 frequencies.");
%% 
% For DTMF 3, the ratio of frequencies is 

High_Frequency = 1477;
Low_Frequency = 697;
DTMF_Ratio_Matrix(1,3) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 3 frequencies.");
%% 
% For DTMF A, the ratio of frequencies is 

High_Frequency = 1633;
Low_Frequency = 697;
DTMF_Ratio_Matrix(1,4) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF A frequencies.");
%% 
% For DTMF 4, the ratio of frequencies is 

High_Frequency = 1209;
Low_Frequency = 770;
DTMF_Ratio_Matrix(2,1) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 4 frequencies.");
%% 
% For DTMF 5, the ratio of frequencies is 

High_Frequency = 1336;
Low_Frequency = 770;
DTMF_Ratio_Matrix(2,2) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 5 frequencies.");
%% 
% For DTMF 6, the ratio of frequencies is 

High_Frequency = 1477;
Low_Frequency = 770;
DTMF_Ratio_Matrix(2,3) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 6 frequencies.");
%% 
% For DTMF B, the ratio of frequencies is 

High_Frequency = 1633;
Low_Frequency = 770;
DTMF_Ratio_Matrix(2,4) = High_Frequency / Low_Frequency; 
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF B frequencies.");
%% 
% For DTMF 7, the ratio of frequencies is 

High_Frequency = 1209;
Low_Frequency = 852;
DTMF_Ratio_Matrix(3,1) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 7 frequencies.");
%% 
% For DTMF 8, the ratio of frequencies is 

High_Frequency = 1336;
Low_Frequency = 852;
DTMF_Ratio_Matrix(3,2) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 8 frequencies.");
%% 
% For DTMF 9, the ratio of frequencies is 

High_Frequency = 1477;
Low_Frequency = 852;
DTMF_Ratio_Matrix(3,3) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 9 frequencies.");
%% 
% For DTMF C, the ratio of frequencies is 

High_Frequency = 1633;
Low_Frequency = 852;
DTMF_Ratio_Matrix(3,4) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF C frequencies.");
%% 
% For DTMF *, the ratio of frequencies is 

High_Frequency = 1209;
Low_Frequency = 941;
DTMF_Ratio_Matrix(4,1) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF * frequencies.");
%% 
% For DTMF 0, the ratio of frequencies is 

High_Frequency = 1336;
Low_Frequency = 941;
DTMF_Ratio_Matrix(4,2) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF 0 frequencies.");
%% 
% For DTMF #, the ratio of frequencies is 

High_Frequency = 1477;
Low_Frequency = 941;
DTMF_Ratio_Matrix(4,3) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF # frequencies.");
%% 
% For DTMF D, the ratio of frequencies is 

High_Frequency = 1633;
Low_Frequency = 941;
DTMF_Ratio_Matrix(4,4) = High_Frequency / Low_Frequency;
disp(num2str(High_Frequency / Low_Frequency) + " is the ratio between DTMF D frequencies.");

disp(DTMF_Ratio_Matrix);
%% 
% From the DTMF Matrix, we could conclude from frequency ratios that the diagonal 
% entries are very close to the linearity; such that DTMF 1, DTMF 5, DTMF 9 and 
% DTMF D.
% 
% If the ratio is smaller, it is harder to separate the frequencies and we need 
% to shrink the window size, or use different techniques enabling detailed analysis 
% due to the inadequate windows.
% 
% If the ratio is bigger, we may use enlarged window sized windows. 
% 
% It could be also seen in part ii that the distance between spectrogram of 
% Pair 1 is closer than Pair 2. However, at Pair 3, the distances are broader 
% than other pairs.
% 
% *Part iv)*
% 
% For DTMF decoding, we may use an STFT-based method including:
%% 
% # Preprocessing, i.e removing the noise after recording the sample
% # Analysis of STFT by appropriate parameters (window size, overlap, FFT size 
% etc.)
% # Detecting the peaks, identifying two frequencies present in each DTMF signal
% # Associating each frequency pair with the corresponding DTMF number.
% # Reconstruction of the sequence.
%% 
% The other methods are, e.g.
%% 
% * Filtering and Tone Detection
% * Using Machine Learning
% * Adaptive Filtering etc.
%% 
% *Functions*

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

function [S, F, T] = mySTFTHamming(signal, fs, windowSize, overlap, nfft)
    % Input:
    %   signal: Input signal
    %   fs: Sampling frequency of the signal
    %   windowSize: Size of the analysis window
    %   overlap: Overlap between consecutive windows (in samples)
    %   nfft: Number of FFT points

    % Calculate the number of overlapping samples
    overlapSamples = overlap;

    % Calculate the number of windows
    numWindows = fix((length(signal) - overlapSamples) / (windowSize - overlapSamples));

    % Initialize the STFT matrix
    S = zeros(nfft/2 + 1, numWindows);

    % Perform the STFT
    for i = 1:numWindows
        % Extract the current window
        startIdx = (i-1)*(windowSize - overlapSamples) + 1;
        endIdx = startIdx + windowSize - 1;
        x = signal(startIdx:endIdx);

        % Apply the Hamming window
        x = x .* hamming(windowSize);

        % Compute the FFT
        X = fft(x, nfft);

        % Store the magnitude spectrum in the STFT matrix
        S(:, i) = abs(X(1:nfft/2 + 1));

        % Store the corresponding time and frequency vectors
        T(i) = startIdx / fs;
        F = linspace(0, fs/2, nfft/2 + 1);
    end
end

function [S, F, T] = mySTFTRectangular(signal, fs, windowSize, overlap, nfft)
    % Input:
    %   signal: Input signal
    %   fs: Sampling frequency of the signal
    %   windowSize: Size of the analysis window
    %   overlap: Overlap between consecutive windows (in samples)
    %   nfft: Number of FFT points

    % Calculate the number of overlapping samples
    overlapSamples = overlap;

    % Calculate the number of windows
    numWindows = fix((length(signal) - overlapSamples) / (windowSize - overlapSamples));

    % Initialize the STFT matrix
    S = zeros(nfft/2 + 1, numWindows);

    % Perform the STFT
    for i = 1:numWindows
        % Extract the current window
        startIdx = (i-1)*(windowSize - overlapSamples) + 1;
        endIdx = startIdx + windowSize - 1;
        x = signal(startIdx:endIdx);

        % Apply the rectangular window
        x = x .* rectwin(windowSize);

        % Compute the FFT
        X = fft(x, nfft);

        % Store the magnitude spectrum in the STFT matrix
        S(:, i) = abs(X(1:nfft/2 + 1));

        % Store the corresponding time and frequency vectors
        T(i) = startIdx / fs;
        F = linspace(0, fs/2, nfft/2 + 1);
    end
end