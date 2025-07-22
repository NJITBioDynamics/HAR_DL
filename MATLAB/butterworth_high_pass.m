function Acc_Filtered = butterworth_high_pass(Acc, fs)
    % Design a 4th order Butterworth high-pass filter with a cutoff frequency of 0.2 Hz
    fc = 0.2; % Cutoff frequency in Hz
    order = 4; % Order of the Butterworth filter

    % Calculate the normalized cutoff frequency (Wn)
    Wn = fc / (fs/2);

    % Design the Butterworth high-pass filter
    [b, a] = butter(order, Wn, 'high');

    % Apply the high-pass filter to the input signal
    Acc_Filtered = filter(b, a, Acc);
end