function Acc_Filtered = butterworth_low_pass(Acc, fs)
    % Design a 4th order Butterworth low-pass filter with a cutoff frequency of 0.2 Hz
    fc = 0.2; % Cutoff frequency in Hz
    order = 4; % Order of the Butterworth filter

    % Calculate the normalized cutoff frequency (Wn)
    Wn = fc / (fs/2);

    % Design the Butterworth low-pass filter
    [b, a] = butter(order, Wn, 'low');

    % Apply the low-pass filter to the input signal
    Acc_Filtered = filter(b, a, Acc);
end
