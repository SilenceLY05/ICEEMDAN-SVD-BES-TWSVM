import os
import numpy as np
import pandas as pd
import time
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from BES1 import initialize_population, select_space, search_space, swoop
from svm import svm

def load_features_from_file(file_path, is_train=True):
    """
        读取CSV文件并提取特征和标签数据。
        对于训练数据，提取特征、标签以及事件信息。
        对于测试数据，提取特征和标签。
        """
    data = pd.read_csv(file_path)

    if is_train:
        features = data.iloc[:, :29].values
        labels = data.iloc[:, -1].values
        max_x_coord = data.iloc[:, -5].values
        event_start = data.iloc[:, -4].values
        event_end = data.iloc[:, -3].values
        event_pos = data.iloc[:, -2].values
        return features, labels, max_x_coord, event_start, event_end, event_pos
    else:
        features = data.iloc[:, :29].values
        labels = data.iloc[:, -1].values
        max_x_coord = data.iloc[:, -2].values
        return features, labels, max_x_coord

def calculate_fitness(individual, X_train, y_train, alpha=0.99):
    """
       计算个体的适应度值。
       根据选择的特征和TWSVM参数进行交叉验证，计算分类准确率和特征选择的比例。
       """
    selected_features = np.where(individual[:29] > 0.5)[0]
    if len(selected_features) == 0:
        return 0  # Prevent division by zero

    c1, c2, gamma = individual[29], individual[30], individual[31]
    fit = svm()
    fit.model = 'twsvm'
    fit.c1 = c1
    fit.c2 = c2
    fit.sigma = gamma

    # 使用交叉验证评估分类器性能
    scores = cross_val_score(fit, X_train[:, selected_features], y_train, cv=5, scoring='accuracy')
    acc = np.mean(scores)

    R = len(selected_features)
    N = X_train.shape[1]

    # 引入惩罚项，当分类准确率很低时，适应度值显著增加
    penalty = 0
    if acc < 0.5:  # 设置一个阈值，比如0.5
        penalty = (0.5 - acc) ** 2

    fitness = alpha * acc + (1 - alpha) * (1 - R / N) + penalty
    return fitness

def evolve_population(pop, BestSol, nPop, low, high, dim, X_train, y_train, MaxIt):
    """
        通过BES算法优化种群，搜索最优的TWSVM参数和特征选择。
        每一轮迭代中，更新种群个体的位置和适应度值。
        """
    Convergence_curve = np.zeros(MaxIt)
    for t in range(MaxIt):
        pop, BestSol, s1 = select_space(lambda ind: calculate_fitness(ind, X_train, y_train), pop, nPop, BestSol, low, high, dim)
        pop, BestSol, s2 = search_space(lambda ind: calculate_fitness(ind, X_train, y_train), pop, BestSol, nPop, low, high)
        pop, BestSol, s3 = swoop(lambda ind: calculate_fitness(ind, X_train, y_train), pop, BestSol, nPop,  low, high)
        Convergence_curve[t] = BestSol['cost']
        print(f"Iteration {t + 1}, Best Cost: {BestSol['cost']}")
    return BestSol, Convergence_curve

def train_and_evaluate(train_folder, test_folder, nPop=100, MaxIt=500):
    """
        读取训练集和测试集数据文件夹中的CSV文件。
        进行TWSVM参数和特征选择的优化。
        在测试数据上评估优化后的模型性能。
        输出准确率、召回率和F1评分。
        """
    train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.csv')]

    best_solutions = []
    # 初始化 feature_low 和 feature_high
    feature_low ,feature_high = None, None

    for file in train_files:
        X_train, y_train, *_ = load_features_from_file(file, is_train=True)

        # 计算每列的最小值和最大值
        if feature_low is None:
            feature_low = np.min(X_train, axis=0)
            feature_high = np.max(X_train, axis=0)
        else:
            feature_low = np.minimum(feature_low, np.min(X_train, axis=0))
            feature_high = np.maximum(feature_high, np.max(X_train, axis=0))

    param_low = np.array([0.01, 0.01, 0.01])
    param_high = np.array([100, 100, 100])

    # 合并特征和参数范围
    low = np.concatenate([feature_low, param_low])
    high = np.concatenate([feature_high, param_high])

    start_time = time.time()
    for file in train_files:
        X_train, y_train, *_ = load_features_from_file(file, is_train=True)
        dim = 32   # 29 features + 3 TWSVM parameters
        pop, BestSol = initialize_population(nPop, dim,  low, high, lambda ind: calculate_fitness(ind, X_train, y_train))
        BestSol, Convergence_curve = evolve_population(pop, BestSol, nPop, low, high, dim, X_train, y_train, MaxIt)
        best_solutions.append(BestSol)

    end_time = time.time()
    timep = end_time - start_time

    all_metrics = []
    for file in test_files:
        X_test, y_test, _ = load_features_from_file(file, is_train=False)
        fit = svm()
        fit.model = 'twsvm'
        selected_features = np.where(best_solutions[0]['pos'][:29] > 0.5)[0]
        fit.c1 = best_solutions[0]['pos'][29]
        fit.c2 = best_solutions[0]['pos'][30]
        fit.sigma = best_solutions[0]['pos'][31]
        y_hat = fit.svm_mc(X_train[:, selected_features], y_train, X_test[:, selected_features], y_test)

        accuracy = np.mean(y_hat == y_test)
        recall = sklearn.metrics.recall_score(y_test, y_hat, average='macro')
        f1 = sklearn.metrics.f1_score(y_test, y_hat, average='macro')

        all_metrics.append((accuracy, recall, f1))
        print(f"Test File: {file}, Accuracy: {accuracy}, Recall: {recall}, F1 Score: {f1}")

    avg_metrics = np.mean(all_metrics, axis=0)
    print(f"Average Accuracy: {avg_metrics[0]}, Average Recall: {avg_metrics[1]}, Average F1 Score: {avg_metrics[2]}")
    print(f"The computation time is: {timep} seconds")

if __name__ == "__main__":
    train_folder = "path_to_train_folder"
    test_folder = "path_to_test_folder"
    train_and_evaluate(train_folder, test_folder)

