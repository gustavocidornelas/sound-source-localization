% parameters used to compute the magnitude and phase spectra
fs = 44100;  % sampling frequency
NFFT = 2^nextpow2(200); 
numrows = NFFT;
nfreq = floor(numrows/2);          % DC is not counted
freqns = (0:nfreq)*(fs/(2*nfreq));
fmin0 = freqns(1);
fmax0 = freqns(end);

% azimuths
azimuths = [-80 -65 -55 -45:5:45 55 65 80];
ITD = zeros(1, 25);
ITD_2 = zeros(1, 25);

for i = 1:25
    % reading the corresponding HRIR
    az = azimuths(i);
    hrir = table2array(readtable(strcat('HRIR/HRIR_Az_', num2str(az), '.csv')));
    hrir_l = hrir(:, 1);
    hrir_r = hrir(:, 2);

    % HRTF for each channel
    hrtf_l_period = fft(hrir_l, NFFT);
    hrtf_l = hrtf_l_period(1:nfreq+1,:); 
    hrtf_r_period = fft(hrir_r, NFFT);
    hrtf_r = hrtf_r_period(1:nfreq+1,:); 
    
    % minimum phase for each channel
    %minp_l = (-imag(hilbert(log(abs(hrtf_l_period)))));
    %minp_r = (-imag(hilbert(log(abs(hrtf_r_period)))));
    
    % finding the mixed-phase signals
    mix_phase_l = unwrap(angle(hrtf_l));
    mix_phase_r = unwrap(angle(hrtf_r));

    % finding the phase of the minimum phase component CHECK THIS
    min_phase_l = unwrap(-imag(hilbert(log(abs(hrtf_l_period)))));
    min_phase_r = unwrap(-imag(hilbert(log(abs(hrtf_r_period)))));

    % computing the excess phase for each channel
    ex_phase_l = (mix_phase_l - min_phase_l(1:nfreq+1,:));
    ex_phase_r = (mix_phase_r - min_phase_r(1:nfreq+1,:));

    
     if i == 1
         plot(freqns ./ 1000, ex_phase_l)
         hold on
         plot(freqns ./ 1000, ex_phase_r)
         %xlim([0 12])
         xlabel('Frequency [kHz]')
         ylabel('Excess phase [rad]')
         grid on
     end

    % fitting a linear model
    lin_model_l = polyfit(freqns(:, 4:6), transpose(ex_phase_l(4:6, :)), 1);
    lin_model_r = polyfit(freqns(:, 4:6), transpose(ex_phase_r(4:6, :)), 1);

    ITD(1, i) = (lin_model_l(1) - lin_model_r(1)) / 44.1 
    
    % method 1: IGD_0
%      % calculating the group delay of the HRTF
%     [gd_hrtf_l, w_l] = grpdelay(hrtf_l);
%     [gd_hrtf_r, w_r] = grpdelay(hrtf_r);
% 
%     % calculating the group delay of the minimum phase component
%     [gd_min_l, w1] = grpdelay(hilbert(log(abs(hrtf_l_period))));
%     [gd_min_r, w2] = grpdelay(hilbert(log(abs(hrtf_r_period))));
% 
%     excess_gd_l = gd_hrtf_l - gd_min_l;
%     excess_gd_r = gd_hrtf_r - gd_min_r;
% 
%     temp = excess_gd_l - excess_gd_r;
%     ITD_2(1, i) = abs(temp(1))

    
end

% saving delays
csvwrite('group_delays.csv', ITD)
%figure()
%plot(azimuths, ITD)
%grid on





%display(0)
