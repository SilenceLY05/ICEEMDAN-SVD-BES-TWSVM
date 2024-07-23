import numpy as np
import matplotlib.pyplot as plt

# 常数 T
T = 500

# 控制曲线平滑度的参数
n = 4  # 可以调整此参数以改变平滑度

# 定义函数
def f(t, T, n):
    return (4/3) - ((4/3) - 1) * (t / T)**n

# 生成 t 值
t = np.linspace(0, T, 1000)
# 计算对应的 f(t) 值
y = f(t, T, n)

# 绘制图像
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title(f'Smooth transition from 4/3 to 1 over T={T}')
plt.grid(True)
plt.show()
