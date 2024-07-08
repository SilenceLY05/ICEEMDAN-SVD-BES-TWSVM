import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from haar import process_signal

# Constants
DIRECTORY_PATH = r'C:\Users\86130\Desktop\新建文件夹'  # Replace with the actual path
WINDOW_SIZE = 50
STEP_SIZE = 20  # New step size for sliding the window
HIGH_SLOPE_THRESHOLD = 2  # 可以根据需要调整阈值
OUTPUT_DIRECTORY = r'C:\Users\86130\Desktop\新建文件夹'  # Set your output directory
LOG_FILE_PATH = r'C:\Users\86130\Desktop\新建文件夹\输出日志.txt'  # 输出日志文件路径

# Get all files from the folder
def get_all_files(directory):
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files if file.endswith('.txt')]
    return paths


# 识别和合并高斜率窗口的功能
def identify_and_merge_high_slope_windows(data, window_size, step_size, threshold, log_file):
    max_slope = 0
    max_slope_position = None
    temp_windows = []

    # 找到最大斜率及其位置
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        X = window_data['X'].values.reshape(-1, 1)
        Y = window_data['Y'].values
        model = LinearRegression()
        model.fit(X, Y)
        slope = model.coef_[0]

        if abs(slope) > max_slope:
            max_slope = abs(slope)
            max_slope_position = window_data['X'].iloc[0]  # 记录最大斜率的起始位置

    # 重新循环以合并高斜率窗口，但仅限于最大斜率位置之前
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        if window_data['X'].iloc[0] > max_slope_position:
            break  # 如果窗口起始位置超过最大斜率位置，停止处理

        X = window_data['X'].values.reshape(-1, 1)
        Y = window_data['Y'].values
        model = LinearRegression()
        model.fit(X, Y)
        slope = model.coef_[0]

        if abs(slope) > threshold:
            temp_windows.append((window_data['X'].iloc[0], window_data['X'].iloc[-1]))

    # 合并高斜率窗口
    high_slope_windows = []
    if temp_windows:
        merged_window = temp_windows[0]
        for start, end in temp_windows[1:]:
            if start <= merged_window[1]:
                merged_window = (merged_window[0], max(end, merged_window[1]))
            else:
                high_slope_windows.append(merged_window)
                merged_window = (start, end)
        high_slope_windows.append(merged_window)

    return high_slope_windows



# Read data from the latest file
#latest_file = get_latest_file(DIRECTORY_PATH)
#if not latest_file:
#    print("No data file found in the specified directory.")
#    exit()

# Assuming the data file is structured as required (two columns separated by a comma)
#data = pd.read_csv(latest_file, header=None)
#data.columns = ['X', 'Y']

# Function to calculate slope and maximum absolute difference within each window and save to a single file
def calculate_and_save_combined_data(file_path, window_size, step_size, output_dir, log_file):
    if file_path is None:
        log_file.write("No data file found.")
        return

    data = pd.read_csv(file_path, header=None)
    data.columns = ['X', 'Y'] 

    # 从第二个程序获取显著点信息
    significant_segments, _ = process_signal(file_path, log_file=log_file)

    # 获取并合并高斜率窗口
    high_slope_windows = identify_and_merge_high_slope_windows(data, window_size, step_size, HIGH_SLOPE_THRESHOLD, log_file)

    combined_data = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        window_start_position = window_data['X'].iloc[0]
        window_end_position = window_data['X'].iloc[-1]

        # Fit a linear regression model to calculate the slope
        X = window_data['X'].values.reshape(-1, 1)
        Y = window_data['Y'].values
        model = LinearRegression()
        model.fit(X, Y)
        slope = model.coef_[0]

        #计算MSE和SMR
        MSE = np.sqrt(np.mean(np.square(Y)))
        SMR = np.sqrt(np.mean(np.abs(Y)))

        # Calculate additional statistics
        abs_mean = np.mean(np.abs(Y))
        average = np.mean(Y)
        maximum = np.max(Y)
        minimum = np.min(Y)
        peak_absolute = np.max(np.abs(Y))
        max_position = window_data['X'].iloc[np.argmax(Y)]

        std_y = np.std(Y)

        # 偏度、峭度、峰-峰值和峰值指标
        S = np.mean(((Y - average) / std_y)**3) if std_y != 0 else 0
        K = np.mean(((Y - average) / std_y)**4) - 3 if std_y != 0 else -3
        PP = np.max(Y) - np.min(Y)
        C = peak_absolute / MSE if MSE != 0 else 0

        WF = MSE / np.abs(average) if average != 0 else 0

        # 裕度指标
        if SMR != 0:
            MF = peak_absolute / SMR  # 裕度指标
        else:
            MF = 0  # 避免除以零错误

        # 新增统计计算：整流平均值 (x_av) 和 变异系数 (Kv)
        x_av = np.mean(np.abs(Y))
        variance = np.var(Y)  # Dx，方差
        if average != 0:
            Kv = np.sqrt(variance) / average
        else:
            Kv = 0  # 如果平均值为零，变异系数设为0，避免除零错误

        # 计算FFT
        fft_values = np.fft.fft(Y)
        frequencies = np.fft.fftfreq(len(fft_values))

        # 取正频率部分
        positive_frequencies = frequencies[:len(frequencies) // 2]
        magnitudes = np.abs(fft_values[:len(fft_values) // 2])  # P(f_k)

        # 频谱中心sepctral centroid
        if np.sum(magnitudes) != 0:
            spectral_centroid = np.sum(positive_frequencies * magnitudes) / np.sum(magnitudes)
        else:
            spectral_centroid = 0

        # 频谱宽度spectral bandwidth
        if np.sum(magnitudes) != 0:
            spectral_bandwidth = np.sqrt(
                np.sum((positive_frequencies - spectral_centroid) ** 2 * magnitudes) / np.sum(magnitudes))
        else:
            spectral_bandwidth = 0

        # MSF
        if np.sum(magnitudes) != 0:
            mean_square_frequency = np.sum(positive_frequencies ** 2 * magnitudes) / np.sum(magnitudes)
        else:
            mean_square_frequency = 0

        # 频谱方差
        if np.sum(magnitudes) != 0:
            frequency_variance = np.sum((positive_frequencies - spectral_centroid) ** 2 * magnitudes) / np.sum(
                magnitudes)
        else:
            frequency_variance = 0

        #均方根频率和频率标准差
        root_mean_square_frequency = np.sqrt(mean_square_frequency)
        standard_deviation_of_frequency = np.sqrt(frequency_variance)

        #频率熵和谱峰稳定指数
        if np.sum(magnitudes) != 0:
            power_normalized = magnitudes / np.sum(magnitudes)  # 归一化功率
        else:
            power_normalized = np.zeros_like(magnitudes)  # 避免除零错误

        spectral_entropy = -np.sum(power_normalized * np.log(power_normalized + 1e-10))  # 防止对0取对数

        positive_frequencies2 = frequencies[1:len(frequencies) // 2]  # 从i=1开始，忽略i=0
        magnitudes2 = np.abs(fft_values[1:len(fft_values) // 2])  # 同上，忽略直流分量
        num = np.sum(positive_frequencies2 ** 2 * magnitudes2)
        num *= np.sum(positive_frequencies2 ** 4 * magnitudes2)
        denom = (np.sum(positive_frequencies2 ** 4 * magnitudes2) * np.sum(positive_frequencies2 ** 2))
        if denom != 0:
            spectral_peak_stability = np.sqrt(num / denom)
        else:
            spectral_peak_stability = 0  # 避免除以零

        # Calculate the absolute difference between consecutive Y-values
        abs_diffs = np.abs(np.diff(Y))
        max_abs_diff_value = np.max(abs_diffs) if abs_diffs.size > 0 else 0
        max_abs_diff_index = np.argmax(abs_diffs) if abs_diffs.size > 0 else -1
        max_abs_diff_position = window_data['X'].iloc[max_abs_diff_index + 1] if max_abs_diff_index != -1 else np.nan

        # 检查显著点信息并添加到结果中
        event_start = event_end = event_peak = None
        #peak_value = None  # 添加变量来存储峰值大小
        label = 0
        for (start_x, end_x, peak_x, peak_val) in significant_segments:
            # 检查显著点是否至少部分位于当前窗口内
            if start_x <= window_end_position and end_x >= window_start_position:
                # 调整事件起始和结束时间，以确保它们在窗口内
                adjusted_start = max(start_x, window_start_position)
                adjusted_end = min(end_x, window_end_position)
                # 检查峰值是否在窗口内
                adjusted_peak = peak_x if window_start_position <= peak_x <= window_end_position else None
                label = 1  # 标记存在显著事件

                # 更新显著事件信息，只有在显著事件确实发生在窗口内时更新
                event_start = adjusted_start
                event_end = adjusted_end
                event_peak = adjusted_peak
                break  # 如果你不希望在找到第一个显著事件后就停止，请移除此break语句

        for (high_start, high_end) in high_slope_windows:
            if high_start <= window_end_position and high_end >= window_start_position:
                adjusted_high_start = max(high_start, window_start_position)
                adjusted_high_end = min(high_end, window_end_position)
                label = 1
                event_start = min(event_start, adjusted_high_start) if event_start else adjusted_high_start
                event_end = max(event_end, adjusted_high_end) if event_end else adjusted_high_end

        # Record the combined information for each window
        combined_data.append([
            window_data['X'].iloc[0], window_data['X'].iloc[-1], slope,
            max_abs_diff_position, max_abs_diff_value,
            abs_mean, average, maximum, minimum, peak_absolute,std_y,
            MSE, SMR, S, K, PP, C,  WF, MF, x_av, Kv, spectral_centroid, spectral_bandwidth,
            mean_square_frequency, frequency_variance, root_mean_square_frequency, standard_deviation_of_frequency,
            spectral_entropy, spectral_peak_stability,max_position, event_start, event_end, event_peak, label#添加新指标
        ])


    # Create a DataFrame and save it to CSV
    output_filename = os.path.basename(file_path).replace('.txt', '_features.csv')
    output_path = os.path.join(output_dir, output_filename)
    combined_df = pd.DataFrame(combined_data, columns=[
        'Start Position', 'End Position', 'Slope',
        'Max Difference Position', 'Max Difference',
        'Absolute Mean', 'Average', 'Maximum', 'Minimum', 'Peak Absolute','Sample Standard',
        'MSE', 'SMR','Skewness', 'Kurtosis', 'Peak-to-Peak', 'Crest Factor',
        'Waveform Factor', 'Margin Factor', 'Rectified Mean', 'Coefficient of Variation',
        'Spectral Centroid', 'Spectral Bandwidth', 'Mean Square Frequency', 'Frequency Variance',
        'Root Mean Square Frequency', 'Standard Deviation of Frequency', 'Spectral Entropy', 'Spectral Peak Stability',
        'Max Position','Event Start', 'Event End', 'Event Position', 'Label'
    ])

    # Create a DataFrame and save it to CSV
    combined_df.to_csv(output_path, index=False)
    log_file.write(f"Data saved to {output_path}")


# Main processing logic
all_files = get_all_files(DIRECTORY_PATH)
with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
    for file_path in all_files:
        calculate_and_save_combined_data(file_path, WINDOW_SIZE, STEP_SIZE, OUTPUT_DIRECTORY, log_file)


"""
# 绘制斜率图
def plot_slope_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    slopes = combined_data['Slope']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, slopes, width=0.1, color='black')
    plt.xlabel('Position (X)')
    plt.ylabel('Slope')
    plt.title('Slope of Best-Fit Line in Each Window')
    plt.grid(True)

# 绘制最大绝对差异图
def plot_max_abs_diff_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    diffs = combined_data['Max Difference']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, diffs, width=0.1, color='black')
    plt.xlabel('Position (X)')
    plt.ylabel('Maximum Absolute Difference')
    plt.title('Maximum Absolute Differences in Each Window')
    plt.grid(True)

# 绘制绝对均值图
def plot_abs_mean_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    abs_means = combined_data['Absolute Mean']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, abs_means, width=0.1, color='green')
    plt.xlabel('Position (X)')
    plt.ylabel('Absolute Mean')
    plt.title('Absolute Means in Each Window')
    plt.grid(True)

# 绘制平均值图
def plot_average_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    averages = combined_data['Average']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, averages, width=0.1, color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Average Value')
    plt.title('Average Values in Each Window')
    plt.grid(True)

# 绘制最大值图
def plot_maximum_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    maximums = combined_data['Maximum']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, maximums, width=0.1, color='red')
    plt.xlabel('Position (X)')
    plt.ylabel('Maximum Value')
    plt.title('Maximum Values in Each Window')
    plt.grid(True)

# 绘制最小值图
def plot_minimum_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    minimums = combined_data['Minimum']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, minimums, width=0.1, color='purple')
    plt.xlabel('Position (X)')
    plt.ylabel('Minimum Value')
    plt.title('Minimum Values in Each Window')
    plt.grid(True)

# 绘制绝对峰值图
def plot_peak_absolute_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    peak_absolutes = combined_data['Peak Absolute']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, peak_absolutes, width=0.1, color='orange')
    plt.xlabel('Position (X)')
    plt.ylabel('Peak Absolute Value')
    plt.title('Peak Absolute Values in Each Window')
    plt.grid(True)

def plot_sample_standard_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    sample_standard = combined_data['Sample Standard']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, sample_standard, width=0.1, color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Sample Standard Value')
    plt.title('Sample Standard Values in Each Window')
    plt.grid(True)

def plot_mse_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    mses = combined_data['MSE']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, mses, width=0.1, color='cyan')
    plt.xlabel('Position (X)')
    plt.ylabel('Mean Square Error (MSE)')
    plt.title('Mean Square Error in Each Window')
    plt.grid(True)

def plot_smr_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    smrs = combined_data['SMR']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, smrs, width=0.1, color='magenta')
    plt.xlabel('Position (X)')
    plt.ylabel('Square Root Amplitude (SMR)')
    plt.title('Square Root Amplitude in Each Window')
    plt.grid(True)

def plot_skewness_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    skewness = combined_data['Skewness']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, skewness, width=0.1, color='grey')
    plt.xlabel('Position (X)')
    plt.ylabel('Skewness')
    plt.title('Skewness in Each Window')
    plt.grid(True)

def plot_kurtosis_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    kurtosis = combined_data['Kurtosis']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, kurtosis, width=0.1, color='brown')
    plt.xlabel('Position (X)')
    plt.ylabel('Kurtosis')
    plt.title('Kurtosis in Each Window')
    plt.grid(True)

def plot_peak_to_peak_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    pp_values = combined_data['Peak-to-Peak']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, pp_values, width=0.1, color='orange')
    plt.xlabel('Position (X)')
    plt.ylabel('Peak-to-Peak')
    plt.title('Peak-to-Peak Values in Each Window')
    plt.grid(True)

def plot_crest_factor_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    crest_factors = combined_data['Crest Factor']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, crest_factors, width=0.1, color='purple')
    plt.xlabel('Position (X)')
    plt.ylabel('Crest Factor')
    plt.title('Crest Factor in Each Window')
    plt.grid(True)

def plot_waveform_factor_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    waveform_factors = combined_data['Waveform Factor']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, waveform_factors, width=0.1, color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Waveform Factor')
    plt.title('Waveform Factor in Each Window')
    plt.grid(True)

def plot_margin_factor_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    margin_factors = combined_data['Margin Factor']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, margin_factors, width=0.1, color='orange')
    plt.xlabel('Position (X)')
    plt.ylabel('Margin Factor')
    plt.title('Margin Factor in Each Window')
    plt.grid(True)

def plot_rectified_mean_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    x_av_values = combined_data['Rectified Mean']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, x_av_values, width=0.1, color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Rectified Mean')
    plt.title('Rectified Mean in Each Window')
    plt.grid(True)

def plot_coefficient_of_variation_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    Kv_values = combined_data['Coefficient of Variation']
    plt.figure(figsize=(12, 6))
    plt.bar(positions, Kv_values, width=0.1, color='green')
    plt.xlabel('Position (X)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Coefficient of Variation in Each Window')
    plt.grid(True)

def plot_spectral_centroid_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    spectral_centroids = combined_data['Spectral Centroid']
    plt.figure(figsize=(12, 6))
    #plt.plot(positions, spectral_centroids, marker='o', linestyle='-', color='blue')
    plt.plot(positions, spectral_centroids, linestyle='-', color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Spectral Centroid')
    plt.title('Spectral Centroid in Each Window')
    plt.grid(True)

def plot_spectral_bandwidth_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    spectral_bandwidths = combined_data['Spectral Bandwidth']
    plt.figure(figsize=(12, 6))
    #plt.plot(positions, spectral_bandwidths, marker='o', linestyle='-', color='red')
    plt.plot(positions, spectral_bandwidths,  linestyle='-', color='red')
    plt.xlabel('Position (X)')
    plt.ylabel('Spectral Bandwidth')
    plt.title('Spectral Bandwidth in Each Window')
    plt.grid(True)

def plot_mean_square_frequency_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    msf_values = combined_data['Mean Square Frequency']
    plt.figure(figsize=(12, 6))
   # plt.plot(positions, msf_values, marker='o', linestyle='-', color='green')
    plt.plot(positions, msf_values, linestyle='-', color='green')
    plt.xlabel('Position (X)')
    plt.ylabel('Mean Square Frequency')
    plt.title('Mean Square Frequency in Each Window')
    plt.grid(True)

def plot_frequency_variance_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    var_f_values = combined_data['Frequency Variance']
    plt.figure(figsize=(12, 6))
    #plt.plot(positions, var_f_values, marker='o', linestyle='-', color='red')
    plt.plot(positions, var_f_values, linestyle='-', color='red')
    plt.xlabel('Position (X)')
    plt.ylabel('Frequency Variance')
    plt.title('Frequency Variance in Each Window')
    plt.grid(True)

def plot_root_mean_square_frequency_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    rmsf_values = combined_data['Root Mean Square Frequency']
    plt.figure(figsize=(12, 6))
    #plt.plot(positions, rmsf_values, marker='o', linestyle='-', color='blue')
    plt.plot(positions, rmsf_values, linestyle='-', color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Root Mean Square Frequency')
    plt.title('Root Mean Square Frequency in Each Window')
    plt.grid(True)

def plot_standard_deviation_of_frequency_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    sigma_f_values = combined_data['Standard Deviation of Frequency']
    plt.figure(figsize=(12, 6))
    #plt.plot(positions, sigma_f_values, marker='o', linestyle='-', color='red')
    plt.plot(positions, sigma_f_values, linestyle='-', color='red')
    plt.xlabel('Position (X)')
    plt.ylabel('Standard Deviation of Frequency')
    plt.title('Standard Deviation of Frequency in Each Window')
    plt.grid(True)

def plot_spectral_entropy_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    entropy_values = combined_data['Spectral Entropy']
    plt.figure(figsize=(12, 6))
   # plt.plot(positions, entropy_values, marker='o', linestyle='-', color='blue')
    plt.plot(positions, entropy_values,linestyle='-', color='blue')
    plt.xlabel('Position (X)')
    plt.ylabel('Spectral Entropy')
    plt.title('Spectral Entropy in Each Window')
    plt.grid(True)

def plot_spectral_peak_stability_graph_from_combined(output_file):
    combined_data = pd.read_csv(output_file)
    positions = (combined_data['Start Position'] + combined_data['End Position']) / 2
    stability_values = combined_data['Spectral Peak Stability']
    plt.figure(figsize=(12, 6))
   # plt.plot(positions, stability_values, marker='o', linestyle='-', color='green')
    plt.plot(positions, stability_values, linestyle='-', color='green')
    plt.xlabel('Position (X)')
    plt.ylabel('Spectral Peak Stability')
    plt.title('Spectral Peak Stability in Each Window')
    plt.grid(True)


# 调用每个绘图函数并显示所有图表
plot_slope_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_max_abs_diff_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_abs_mean_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_average_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_maximum_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_minimum_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_peak_absolute_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_mse_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_smr_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_skewness_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_kurtosis_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_peak_to_peak_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_crest_factor_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_waveform_factor_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_margin_factor_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_rectified_mean_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_coefficient_of_variation_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_spectral_centroid_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_spectral_bandwidth_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_mean_square_frequency_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_frequency_variance_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_root_mean_square_frequency_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_standard_deviation_of_frequency_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_spectral_entropy_graph_from_combined(OUTPUT_CSV)
plt.show()

plot_spectral_peak_stability_graph_from_combined(OUTPUT_CSV)
plt.show()

"""