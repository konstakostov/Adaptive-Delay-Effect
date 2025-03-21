import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt


def compute_rms_envelope(x, block_size):
    """
    Compute RMS (Root Mean Square) values for non-overlapping blocks.

    Parameters:
        x (ndarray): Input audio signal
        block_size (int): Number of samples per analysis block

    Returns:
        ndarray: Array of RMS values, one per block

    Notes:
        RMS represents the effective power of the signal in each block
    """
    num_blocks = len(x) // block_size
    rms_values = np.zeros(num_blocks)

    for i in range(num_blocks):
        start = i * block_size
        frame = x[start:start + block_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))

    return rms_values


def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Apply a Butterworth lowpass filter to smooth data.

    Parameters:
        data (ndarray): Signal to be filtered
        cutoff_freq (float): Cutoff frequency in Hz
        sample_rate (float): Sample rate of signal in Hz
        order (int): Filter order, controls steepness of cutoff

    Returns:
        ndarray: Filtered signal

    Notes:
        Higher order values create steeper cutoffs but may introduce ringing
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)[:2]

    return filtfilt(b, a, data)


def adaptive_basic_delay(x, sample_rate, block_size, base_delay_sec, max_delay_sec, gain, cutoff_freq=1.0):
    """
    Apply an adaptive delay effect where delay time varies based on signal amplitude.

    Parameters:
        x (ndarray): Input audio signal (mono)
        sample_rate (float): Sample rate in Hz
        block_size (int): Analysis block size in samples
        base_delay_sec (float): Minimum delay time in seconds
        max_delay_sec (float): Maximum delay time in seconds
        gain (float): Gain factor for the delayed signal
        cutoff_freq (float): Cutoff frequency for envelope smoothing in Hz

    Returns:
        ndarray: Processed audio with adaptive delay effect

    Algorithm:
        1. Compute RMS envelope from non-overlapping blocks
        2. Smooth envelope with lowpass filter
        3. Normalize RMS values to 0-1 range
        4. Map normalized values to delay times (higher RMS = shorter delay)
        5. Apply sample-by-sample delay processing

    Notes:
        Output is extended to accommodate maximum delay time
    """
    # Compute the RMS envelope for the whole signal.
    rms_env = compute_rms_envelope(x, block_size)

    # Apply Butterworth low-pass filter to smooth the envelope
    rms_env = lowpass_filter(rms_env, cutoff_freq, sample_rate, order=4)

    rms_min = np.min(rms_env)
    rms_max = (np.max(rms_env) + 1e-6)  # Avoid division by zero

    # Determine the maximum delay in samples for buffer extension.
    max_delay_samples = int(np.ceil(max_delay_sec * sample_rate))
    output_length = len(x) + max_delay_samples
    y = np.zeros(output_length, dtype=x.dtype)

    # Copy dry signal into output.
    y[:len(x)] = x

    num_blocks = len(rms_env)

    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size

        # Compute normalized RMS for this block:
        norm_rms = (rms_env[i] - rms_min) / (rms_max - rms_min)

        # Map to delay time: higher RMS --> shorter delay.
        delay_time = base_delay_sec + (max_delay_sec - base_delay_sec) * (1 - norm_rms)
        delay_samples = int(np.ceil(delay_time * sample_rate))

        # Process each sample in this block:
        for n in range(block_start, min(block_end, len(x))):
            if n >= delay_samples:
                y[n + delay_samples] += gain * x[n - delay_samples]

    return y


def plot_waveforms(original, processed, sample_rate, output_filename="waveform_comparison.png"):
    """
    Generate comparison plots of original and processed waveforms.

    Parameters:
        original (ndarray): Original audio signal
        processed (ndarray): Processed audio signal
        sample_rate (float): Sample rate in Hz
        output_filename (str): Path to save the output image

    Outputs:
        Four subplots:
        1. Original signal waveform
        2. Processed signal waveform
        3. Overlay of both signals
        4. Difference between signals

    Notes:
        Automatically saves the plot to the specified file
    """
    time_axis = np.linspace(0, len(original) / sample_rate, num=len(original))

    difference = processed[:len(original)] - original  # Ensure same length for difference calculation

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time_axis, original, label="Original Signal", color="blue")
    axes[0].set_title("Original Audio Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(time_axis, processed[:len(original)], label="Processed Signal", color="green")
    axes[1].set_title("Processed Audio Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(time_axis, original, label="Original Signal", color="blue", alpha=0.6)
    axes[2].plot(time_axis, processed[:len(original)], label="Processed Signal", color="gray", alpha=0.6)
    axes[2].set_title("Overlay: Processed vs. Original")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(time_axis, difference, label="Difference (Processed - Original)", color="red")
    axes[3].set_title("Difference Between Signals")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Waveform plot saved as {output_filename}")


def plot_spectrogram(original, processed, sample_rate, output_filename="spectrogram_comparison.png"):
    """
    Generate comparison spectrogram of original and processed signals.

    Parameters:
        original (ndarray): Original audio signal
        processed (ndarray): Processed audio signal
        sample_rate (float): Sample rate in Hz
        output_filename (str): Path to save the output image

    Outputs:
        Three subplots:
        1. Original signal spectrogram
        2. Processed signal spectrogram
        3. Difference spectrogram

    Notes:
        Uses logarithmic scaling (dB) with small epsilon to avoid log(0)
        Automatically saves the plot to the specified file
    """
    f1, t1, Sxx_orig = spectrogram(original, sample_rate)
    f2, t2, Sxx_proc = spectrogram(processed[:len(original)], sample_rate)

    difference = Sxx_proc - Sxx_orig  # Difference spectrogram

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Add a small constant to avoid log of zero
    epsilon = 1e-10

    im1 = axes[0].pcolormesh(t1, f1, 10 * np.log10(Sxx_orig + epsilon), shading='auto')
    axes[0].set_title("Original Signal Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[0])
    axes[0].grid(True)

    im2 = axes[1].pcolormesh(t2, f2, 10 * np.log10(Sxx_proc + epsilon), shading='auto')
    axes[1].set_title("Processed Signal Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=axes[1])
    axes[1].grid(True)

    im3 = axes[2].pcolormesh(t1, f1, 10 * np.log10(np.abs(difference) + epsilon), shading='auto', cmap="coolwarm")
    axes[2].set_title("Difference Spectrogram (Processed - Original)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (seconds)")
    fig.colorbar(im3, ax=axes[2])
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

    print(f"Spectrogram plot saved as {output_filename}")


# Add this to main.py
def process_audio_with_adaptive_delay(
        input_signal,
        sampling_rate,
        output_filename,
        frame_size=1024,
        min_delay=0.1,
        max_delay=1.0,
        gain_delay=0.5,
        cutoff_frequency=1.0,
        waveform_plot=None,
        spectrogram_plot=None
):
    """
    Process audio signal with adaptive delay effect and save the result.

    Parameters:
        input_signal (ndarray): Input audio signal
        sampling_rate (float): Sample rate in Hz
        output_filename (str): Path to save the processed audio
        frame_size (int): Block size in samples for RMS analysis
        min_delay (float): Minimum delay time (seconds) - applied during high amplitude
        max_delay (float): Maximum delay time (seconds) - applied during low amplitude
        gain_delay (float): Gain for delayed signal - controls feedback intensity
        cutoff_frequency (float): Cutoff frequency (Hz) for envelope smoothing
        waveform_plot (str, optional): Path to save waveform comparison plot
        spectrogram_plot (str, optional): Path to save spectrogram comparison plot

    Returns:
        ndarray: Processed audio signal
    """
    # Apply the adaptive delay effect
    output_signal = adaptive_basic_delay(
        input_signal,
        sampling_rate,
        frame_size,
        min_delay,
        max_delay,
        gain_delay,
        cutoff_frequency
    )

    # Save the processed output
    sf.write(output_filename, output_signal, sampling_rate)
    print(f"Processed signal saved to {output_filename}")

    # Generate visual representations if paths are provided
    if waveform_plot:
        plot_waveforms(input_signal, output_signal, sampling_rate, waveform_plot)

    if spectrogram_plot:
        plot_spectrogram(input_signal, output_signal, sampling_rate, spectrogram_plot)

    return output_signal