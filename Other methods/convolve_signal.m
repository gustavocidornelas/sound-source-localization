
[audio_signal, fs] = audioread('NFIMenglish.wav');

% azimuths
azimuths = [-80 -65 -55 -45:5:45 55 65 80];

% convolving the audio signal with each HRIR to save the audio files
for i = 1:25
    az = azimuths(i);
    hrir = table2array(readtable(strcat('HRIR_Az_', num2str(az), '.csv')));
   
    % separating the impulse responses from the left and from the right
    hrir_l = hrir(:, 1);
    hrir_r = hrir(:, 2);
    
    % obtaining signals for the left and right channels
    signal_l = conv(audio_signal, hrir_l);
    signal_r = conv(audio_signal, hrir_r);
    
    % saving the signals to the corresponding csv file
    csvwrite(strcat('Speech_Az_', num2str(az), '.csv'), cat(2, signal_l, signal_r))
end
