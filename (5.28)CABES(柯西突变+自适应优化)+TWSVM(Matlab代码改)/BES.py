import numpy as np
import time

def BES(nPop, MaxIt, low, high, dim, fobj):
    """
    秃鹰搜索优化算法
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
    start_time = time.time()

    # 初始化最佳解决方案
    BestSol = {'cost': float('inf')}
    pop = {'pos': np.zeros((nPop, dim)), 'cost': np.zeros(nPop)}

    # 初始化种群
    for i in range(nPop):
        pop['pos'][i, :] = low + (high - low) * np.random.rand(dim)  #为第i个个体生成随机位置
        pop['cost'][i] = fobj(pop['pos'][i, :])                      #计算第i个个体的适应度值
        if pop['cost'][i] < BestSol['cost']:
            BestSol['pos'] = pop['pos'][i, :].copy()                #更新最佳解决方案的位置为当前个体的位置。
            BestSol['cost'] = pop['cost'][i]                        #更新最佳解决方案的适应度值为当前个体的适应度值。

    print(f"0 {BestSol['cost']}")

    Convergence_curve = np.zeros(MaxIt)

    # 算法主循环
    for t in range(MaxIt):
        # 1. 选择空间
        pop, BestSol, s1 = select_space(fobj, pop, nPop, BestSol, low, high, dim)
        # 2. 在空间中搜索
        pop, BestSol, s2 = search_space(fobj, pop, BestSol, nPop, low, high)
        # 3. 俯冲
        pop, BestSol, s3 = swoop(fobj, pop, BestSol, nPop, low, high)

        Convergence_curve[t] = BestSol['cost']
        print(f"{t + 1} {BestSol['cost']}")

    end_time = time.time()
    timep = end_time - start_time

    return BestSol, Convergence_curve, timep

def select_space(fobj, pop, npop, BestSol, low, high, dim):
    """
    选择空间阶段
    参数:
    fobj : function  目标函数，用于计算每个个体的适应度值。
    pop : dict  当前种群。
    npop : int  种群规模。
    BestSol : dict  当前最佳解决方案。
    low : float  决策变量的下界。
    high : float  决策变量的上界。
    dim : int  决策变量的维度。

    返回:
    pop : dict  更新后的种群。
    BestSol : dict  更新后的最佳解决方案。
    s1 : int  记录成功更新的个体数量。
    """
    Mean = np.mean(pop['pos'], axis=0)
    lm = 2
    s1 = 0

    for i in range(npop):
        newsol_pos = BestSol['pos'] + lm * np.random.rand(dim) * (Mean - pop['pos'][i, :])
        newsol_pos = np.clip(newsol_pos, low, high)               #确保生成的新候选解位置在指定的范围内
        newsol_cost = fobj(newsol_pos)

        if newsol_cost < pop['cost'][i]:                          #检查新候选解的适度值是否优于第i个个体的
            pop['pos'][i, :] = newsol_pos                         #更新第 i 个个体的位置为新候选解的位置 newsol_pos。
            pop['cost'][i] = newsol_cost                          #更新第 i 个个体的适应度值为新候选解的适应度值 newsol_cost。
            s1 += 1
            if newsol_cost < BestSol['cost']:                     #检查新候选解 newsol_pos 的适应度值 newsol_cost 是否优于当前最佳解决方案的适应度值
                BestSol['pos'] = newsol_pos.copy()                #更新最佳解决方案的位置为新候选解的位置 newsol_pos。
                BestSol['cost'] = newsol_cost                     #更新最佳解决方案的适应度值为新候选解的适应度值 newsol_cost。

    return pop, BestSol, s1

def search_space(fobj, pop, BestSol, npop, low, high):
    """
    在空间中搜索阶段
    参数:
    fobj : function  目标函数，用于计算每个个体的适应度值。
    pop : dict  当前种群。
    BestSol : dict  当前最佳解决方案。
    npop : int  种群规模。
    low : float  决策变量的下界。
    high : float  决策变量的上界。

    返回:
    pop : dict  更新后的种群。
    BestSol : dict  更新后的最佳解决方案。
    s1 : int  记录成功更新的个体数量。
    """
    Mean = np.mean(pop['pos'], axis=0)
    a = 10
    R = 1.5
    s1 = 0

    for i in range(npop - 1):
        #A = np.random.permutation(npop)         #生成一个长度为npop的随机排列数组
        #pop['pos'] = pop['pos'][A, :]           #将种群的位置和适应度值按A中的顺序重新排列
        #pop['cost'] = pop['cost'][A]

        x, y = polr(a, R, npop)
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

def swoop(fobj, pop, BestSol, npop, low, high):
    """
    俯冲阶段
    参数:
    fobj : function  目标函数，用于计算每个个体的适应度值。
    pop : dict  当前种群。
    BestSol : dict  当前最佳解决方案。
    npop : int  种群规模。
    low : float  决策变量的下界。
    high : float  决策变量的上界。

    返回:
    pop : dict  更新后的种群。
    BestSol : dict  更新后的最佳解决方案。
    s1 : int  记录成功更新的个体数量。
    """
    Mean = np.mean(pop['pos'], axis=0)
    a = 10
    R = 1.5
    s1 = 0

    for i in range(npop):
        #A = np.random.permutation(npop)
        #pop['pos'] = pop['pos'][A, :]
        #pop['cost'] = pop['cost'][A]

        x, y = swoo_p(a, R, npop)
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

def swoo_p(a, R, N):
    """
    生成俯冲阶段所需的极坐标
    参数:
    a : float  参数a，用于计算角度。
    R : float  参数R，用于计算半径。
    N : int  数量。

    返回:
    xR : numpy.ndarray  x坐标数组。
    yR : numpy.ndarray  y坐标数组。
    """
    th = a * np.pi * np.exp(np.random.rand(N))
    r = th  # R * np.random.rand(N)
    xR = r * np.sinh(th)
    yR = r * np.cosh(th)
    xR = xR / np.max(np.abs(xR))
    yR = yR / np.max(np.abs(yR))
    return xR, yR

def polr(a, R, N):
    """
    生成搜索阶段所需的极坐标
    参数:a : float  参数a，用于计算角度。
    R : float  参数R，用于计算半径。
    N : int  数量。

    返回:
    xR : numpy.ndarray  x坐标数组。
    yR : numpy.ndarray  y坐标数组。
    """
    th = a * np.pi * np.random.rand(N)
    r = th + R * np.random.rand(N)
    xR = r * np.sin(th)
    yR = r * np.cos(th)
    xR = xR / np.max(np.abs(xR))
    yR = yR / np.max(np.abs(yR))
    return xR, yR
