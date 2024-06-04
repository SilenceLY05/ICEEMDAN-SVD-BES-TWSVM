
import numpy as np

def calculate_snr(original_signal, denoised_signal):
    """
    Calculate Signal-to-Noise Ratio (SNR) of the denoised signal compared to the original.
    Parameters:
    - original_signal: Original clean signal.
    - denoised_signal: Signal after denoising.
    Returns:
    - SNR in decibels (dB).
    """
    # Calculate the power of the original signal
    signal_power = np.sum(original_signal ** 2) / len(original_signal)
    # Calculate the power of the noise (difference between original and denoised signals)
    noise_power = np.sum((original_signal - denoised_signal) ** 2) / len(original_signal)
    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(original_signal, denoised_signal):
    """
    Calculate Root Mean Square Error (RMSE) of the denoised signal compared to the original.
    Parameters:
    - original_signal: Original clean signal.
    - denoised_signal: Signal after denoising.
    Returns:
    - RMSE.
    """
    # Calculate RMSE
    rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
    return rmse
