import pandas as pd
import matplotlib.pyplot as plt

# 替换这里的文件路径为您的Excel文件路径
file_path = ('G:\毕业论文文件\OTDR_Data\Before treatment\pw50_samp80_lp200_av62_sens1000_Pos_9185_i4960_SigN_708466_PM_995606_PNR_1.4.csv')

# 读取Excel文件
data = pd.read_csv(file_path, sep=';', header=None, names=['Distance (m)', 'Power (dB)'])

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(data['Distance (m)'], data['Power (dB)'], marker='o', linestyle='-', color='blue')
plt.title('Distance vs. Power')
plt.xlabel('Distance (m)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()
