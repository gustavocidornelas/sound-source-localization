% loading the corresponding HRIR file
load('hrir_final.mat');

% selecting the HRIR for the horizontal plane
HRIR_l = reshape(hrir_l(:, 9, :), [25, 200]);
HRIR_r = reshape(hrir_r(:, 9, :), [25, 200]);

% visualizing some impulse responses
fs = 44100.0;
time = (0:(1/fs):((200-1)/fs))*1000;  % time in msec
plot(time, HRIR_l(10, :))
hold on
plot(time, HRIR_r(10, :))

% azimuths
azimuths = [-80 -65 -55 -45:5:45 55 65 80];

% saving all impulse responses properly
for i = 1:25
    curr_hrir_l = reshape(HRIR_l(i, :), [200, 1]);
    curr_hrir_r = reshape(HRIR_r(i, :), [200, 1]);
    az = azimuths(i);
    
    csvwrite(strcat('HRIR_Az_', num2str(az), '.csv'), cat(2, curr_hrir_l, curr_hrir_r))
    
end
