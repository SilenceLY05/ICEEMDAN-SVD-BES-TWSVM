import numpy as np
import pywt
import matplotlib.pyplot as plt

def plot_wavelet_decomposition(signal, wavelet_name, level=5):
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    plt.figure(figsize=(10, 8))
    plt.subplot(level + 2, 1, 1)
    plt.title(f'Original Signal - {wavelet_name}')
    plt.plot(signal)
    for i, coef in enumerate(coeffs):
        plt.subplot(level + 2, 1, i + 2)
        plt.plot(coef, label=f'Level {level - i}')
        plt.legend()
    plt.tight_layout()
    plt.show()

def read_signal_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Skip the first line and parse the rest
    data = np.array([list(map(float, line.strip().split(','))) for line in lines[1:]])
    return data[:, 1]  # Assuming the signal is in the second column

# List of wavelets to be used for decomposition
wavelet_names = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'haar', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'coif5', 'bior3.1', 'dmey']

# Reading signal
signal = read_signal_from_file(r'G:\毕业论文文件\OTDR_Data\After treatment\122\processed_data.txt')

# Checking if signal is long enough for 5 levels of decomposition
min_length = 2 ** 5  # Minimum length needed for 5 levels of decomposition
if len(signal) >= min_length:
    # Plotting decomposition for each wavelet
    for wavelet_name in wavelet_names:
        plot_wavelet_decomposition(signal, wavelet_name, level=5)
else:
    print(f"Signal length is too short for 5 levels of decomposition. Length needed: {min_length}, actual length: {len(signal)}")
