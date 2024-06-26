import numpy as np
import time

def initialize_population(nPop, dim, low, high, fobj):
    """
    初始化种群
    参数:
    nPop : int  种群规模。
    dim : int  决策变量的维度。
    low : float  决策变量的下界。
    high : float  决策变量的上界。
    fobj : function  目标函数，用于计算每个个体的适应度值。

    返回:
    pop : dict  包含种群位置和适应度值的字典。
    BestSol : dict  包含最佳解决方案位置及其代价的字典。
    """
    pop = {'pos': np.zeros((nPop, dim)), 'cost': np.zeros(nPop)}
    BestSol = {'cost': float('inf')}

    for i in range(nPop):
        # 初始化特征选择部分
        pop['pos'][i, :29] = low[:29] + (high[:29] - low[:29]) * np.random.rand(29)
        # 初始化TWSVM参数部分
        pop['pos'][i, 29:] = low[29:] + (high[29:] - low[29:]) * np.random.rand(3)
        # 计算适应度值
        pop['cost'][i] = fobj(pop['pos'][i, :])
        # 更新最佳解决方案
        if pop['cost'][i] < BestSol['cost']:
            BestSol['pos'] = pop['pos'][i, :].copy()
            BestSol['cost'] = pop['cost'][i]

    return pop, BestSol

def select_space(fobj, pop, npop, BestSol, low, high, dim):
    Mean = np.mean(pop['pos'], axis=0)
    lm = 2
    s1 = 0

    for i in range(npop):
        newsol_pos = BestSol['pos'] + lm * np.random.rand(dim) * (Mean - pop['pos'][i, :])
        newsol_pos = np.clip(newsol_pos, low, high)         #确保生成的新候选解位置在指定的范围内
        newsol_cost = fobj(newsol_pos)

        if newsol_cost < pop['cost'][i]:
            pop['pos'][i, :] = newsol_pos
            pop['cost'][i] = newsol_cost
            s1 += 1
            if newsol_cost < BestSol['cost']:
                BestSol['pos'] = newsol_pos.copy()
                BestSol['cost'] = newsol_cost

    return pop, BestSol, s1

def search_space(fobj, pop, BestSol, npop, low, high,th):
    Mean = np.mean(pop['pos'], axis=0)
    a = 10
    s1 = 0

    for i in range(npop - 1):
        x, y = polr(a, npop,th)
        Step = pop['pos'][i, :] - pop['pos'][i + 1, :]
        Step1 = pop['pos'][i, :] - Mean
        newsol_pos = pop['pos'][i, :] + y[i] * Step + x[i] * Step1
        newsol_pos = np.clip(newsol_pos, low, high)
        newsol_cost = fobj(newsol_pos)

        if newsol_cost < pop['cost'][i]:
            pop['pos'][i, :] = newsol_pos
            pop['cost'][i] = newsol_cost
            s1 += 1
            if newsol_cost < BestSol['cost']:
                BestSol['pos'] = newsol_pos.copy()
                BestSol['cost'] = newsol_cost

    return pop, BestSol, s1

def swoop(fobj, pop, BestSol, npop, low, high, th):
    Mean = np.mean(pop['pos'], axis=0)
    a = 10
    s1 = 0

    for i in range(npop):
        x, y = swoo_p(a,  npop, th)
        Step = pop['pos'][i, :] - 2 * Mean
        Step1 = pop['pos'][i, :] - 2 * BestSol['pos']
        newsol_pos = np.random.rand(len(Mean)) * BestSol['pos'] + x[i] * Step + y[i] * Step1
        newsol_pos = np.clip(newsol_pos, low, high)
        newsol_cost = fobj(newsol_pos)

        if newsol_cost < pop['cost'][i]:
            pop['pos'][i, :] = newsol_pos
            pop['cost'][i] = newsol_cost
            s1 += 1
            if newsol_cost < BestSol['cost']:
                BestSol['pos'] = newsol_pos.copy()
                BestSol['cost'] = newsol_cost

    return pop, BestSol, s1

def polr(a, N, th):
    r = a * np.random.rand(N) * np.pi
    omega1 = r * np.sin(th)
    eta1 = r * np.cos(th)
    m1 = omega1 / np.max(np.abs(omega1))
    n1 = eta1 / np.max(eta1)
    return m1, n1

def swoo_p(a, N,th):
    r = a * np.random.rand(N) * np.pi
    omega2 = r * np.sinh(th)
    eta2 = r * np.cosh(th)
    m2 = omega2 / np.max(np.abs(omega2))
    n2 = eta2 / np.max(np.abs(eta2))
    return m2, n2


"""
def run_BES(nPop, MaxIt, low, high, dim, fobj):
    """
"""
    运行秃鹰搜索优化算法
    参数:
    nPop : int  种群规模。
    MaxIt : int  最大迭代次数。
    low : float  决策变量的下界。
    high : float  决策变量的上界。
    dim : int  决策变量的维度。
    fobj : function  目标函数，用于计算每个个体的适应度值。

    返回:
    BestSol : dict  包含最佳解决方案位置及其代价的字典。
    Convergence_curve : numpy.ndarray  每次迭代时最佳代价值的数组。
    timep : float  总计算时间。
    """
"""
    start_time = time.time()

    # 初始化种群和最佳解决方案
    pop, BestSol = initialize_population(nPop, dim, low, high, fobj)
    print(f"0 {BestSol['cost']}")

    Convergence_curve = np.zeros(MaxIt)

    # 算法主循环
    for t in range(MaxIt):
        pop, BestSol, s1 = select_space(fobj, pop, nPop, BestSol, low, high, dim)
        pop, BestSol, s2 = search_space(fobj, pop, BestSol, nPop, low, high)
        pop, BestSol, s3 = swoop(fobj, pop, BestSol, nPop, low, high)

        Convergence_curve[t] = BestSol['cost']
        print(f"{t + 1} {BestSol['cost']}")

    end_time = time.time()
    timep = end_time - start_time

    return BestSol, Convergence_curve, timep
"""


