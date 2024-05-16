import numpy as np
from math import factorial
from itertools import permutations


def permutation_entropy(time_series, m, tau):
    """计算给定时间序列的排列熵。

    参数:
        time_series (list): 输入的时间序列。
        m (int): 嵌入维度。
        tau (int): 时间延迟。

    返回:
        float: 排列熵值。
    """
    n = len(time_series)
    if n <= 0 or m * tau > n:
        raise ValueError("时间序列长度不足以计算排列熵。")

    # 计算所有可能的排列
    perm = list(permutations(range(m)))
    c = {p: 0 for p in perm}  # 初始化计数器

    # 生成所有可能的 m 维向量并计算排列
    for i in range(n - tau * (m - 1)):
        # 提取元素并基于大小进行排列
        sorted_index_tuple = tuple(np.argsort(time_series[i:i + tau * m:tau]))
        c[sorted_index_tuple] += 1

    # 计算概率分布
    probabilities = np.array(list(c.values())) / (n - tau * (m - 1))

    # 计算熵
    entropy = -np.sum(prob * np.log(prob) for prob in probabilities if prob > 0)
    return entropy