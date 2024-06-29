import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from SABES import BaseSABES
from svm import svm


# 读取特征数据文件
def load_features_from_file(filepath, is_training=True):
    data = pd.read_csv(filepath)
    if is_training:
        features = data.iloc[:, :-5]  # 取前29列作为特征
        labels = data.iloc[:, -1]  # 最后一列作为标签
        max_x_coords = data.iloc[:, -5]  # 窗口内最大值横坐标
        events_info = data.iloc[:, -4:-1]  # 事件信息（事件起始点、事件结束点、事件位置）
    else:
        features = data.iloc[:, :-2]  # 取前29列作为特征（测试集没有事件信息）
        labels = data.iloc[:, -1]  # 最后一列作为标签
        max_x_coords = data.iloc[:, -2]  # 窗口内最大值横坐标
        events_info = None  # 测试集没有事件信息
    return features.values, labels.values, max_x_coords.values, events_info.values if events_info is not None else None


# 初始化秃鹰种群和TWSVM参数，包括特征向量选择和TWSVM的参数C1、C2和γ
def initialize_population(pop_size, feature_dim):
    population = []
    for _ in range(pop_size):
        individual = {
            'features': np.random.choice([0, 1], size=feature_dim),  # 特征选择二进制向量
            'C1': np.random.uniform(0.1, 10),
            'C2': np.random.uniform(0.1, 10),
            'gamma': np.random.uniform(0.1, 10),
            'fitness': None
        }
        population.append(individual)
    return population



# 计算秃鹰种群个体的适应度值
def calculate_fitness(individual, X, y, fitness_alpha, beta):
    selected_features = np.where(individual['features'] == 1)[0]
    if len(selected_features) == 0:
        return 0
    X_selected = X[:, selected_features]
    model = svm(kernel='rbf')
    model.model = 'twsvm'
    model.method = 'ovo'
    model.c1 = individual['C1']
    model.c2 = individual['C2']
    model.sigma = individual['gamma']
    acc = np.mean(cross_val_score(model, X_selected, y, cv=5))
    R = len(selected_features)
    N = X.shape[1]
    fitness = fitness_alpha * acc + beta * (1 - R / N)
    return fitness


# 利用BES算法搜索TWSVM参数C1、C2、γ和特征
def evolve_population(population, X, y, fitness_alpha, beta):
    for individual in population:
        individual['fitness'] = calculate_fitness(individual, X, y, fitness_alpha, beta)
    best_individual = max(population, key=lambda ind: ind['fitness'])
    return population, best_individual


# 主循环：优化TWSVM参数和特征选择
def optimize_with_bes(X, y, max_epochs=100, pop_size=20, fitness_alpha=0.5):
    beta = 1 - fitness_alpha
    feature_dim = X.shape[1]

    # 初始化种群
    population = initialize_population(pop_size, feature_dim)

    # 进化种群
    optimizer = BaseSABES(problem={
        "fit_func": lambda individual: -calculate_fitness(individual, X, y, fitness_alpha, beta),
        "lb": [0] * feature_dim + [0.1, 0.1, 0.1],
        "ub": [1] * feature_dim + [10, 10, 10],
        "minmax": "min",
    }, epoch=max_epochs, pop_size=pop_size, a=10, beta=1.2, alpha=1.5, c1=2, c2=2)

    best_individual = None
    convergence_curve = []  # 记录每次迭代的最佳适应度值

    for epoch in range(max_epochs):
        optimizer.evolve(epoch)
        population, current_best = evolve_population(optimizer.pop, X, y, fitness_alpha, beta)
        current_best = max(optimizer.pop, key=lambda ind: ind['fitness'])
        if best_individual is None or current_best['fitness'] > best_individual['fitness']:
            best_individual = current_best
        convergence_curve.append(best_individual['fitness'])
        print(f'Epoch {epoch + 1}/{max_epochs}, Best Fitness: {best_individual["fitness"]:.4f}')

    return best_individual, convergence_curve


# 生成事件信息
def generate_event_info(y_pred, max_x_coords, window_size=10):
    events_info = []
    event_start, event_end, event_position = None, None, None
    max_y_value, max_y_index = None, None

    for i, label in enumerate(y_pred):
        if label in [1, 2, 3, 4, 5]:  # 事件标签
            if event_start is None:
                event_start = i * window_size  # 窗口起始点
            event_end = (i + 1) * window_size  # 窗口结束点

            # 更新事件范围内的最大值和位置
            current_max_x_coord = max_x_coords[i]
            if max_y_value is None or current_max_x_coord > max_y_value:
                max_y_value = current_max_x_coord
                event_position = current_max_x_coord

        else:
            if event_start is not None:
                events_info.append([event_start, event_end, event_position])
                event_start, event_end, max_y_value, event_position = None, None, None, None

    if event_start is not None:
        events_info.append([event_start, event_end, event_position])

    return events_info

# 评估模型
def evaluate_model(best_individual, X_train, y_train, X_test, y_test, max_x_coords):
    selected_features = np.where(best_individual['features'] == 1)[0]
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    model = svm(kernel='rbf')
    model.model = 'twsvm'
    model.method = 'ovo'
    model.c1 = best_individual['C1']
    model.c2 = best_individual['C2']
    model.sigma = best_individual['gamma']

    #训练模型
    model.train(X_train_selected, y_train, 'twsvm')

    # 对测试集进行分类预测
    y_pred = model.predict(X_test_selected, X_train_selected, y_train)

    #计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 生成事件信息
    events_info = generate_event_info(y_pred,max_x_coords)

    return accuracy, recall, f1, events_info

def plot_convergence_curve(convergence_curve):
    plt.plot(convergence_curve)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.show()


# 训练和评估模型
def train_and_evaluate(train_dir, test_dir, fitness_alpha=0.5, max_epochs=100, pop_size=20):
    best_individual = None

    # 训练模型
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(train_dir, filename)
            X_train, y_train, _, _ = load_features_from_file(filepath, is_training=True)

            # 优化TWSVM参数和特征选择
            best_individual, convergence_curve = optimize_with_bes(X_train, y_train, max_epochs, pop_size, fitness_alpha)

    all_accuracy, all_recall, all_f1 = [], [], []
    all_events_info = []

    # 测试模型
    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(test_dir, filename)
            X_test, y_test, max_x_coords, _ = load_features_from_file(filepath, is_training=False)

            # 评估模型
            accuracy, recall, f1, events_info = evaluate_model(best_individual, X_train, y_train, X_test, y_test, max_x_coords)
            all_accuracy.append(accuracy)
            all_recall.append(recall)
            all_f1.append(f1)
            all_events_info.extend(events_info)

    avg_accuracy = np.mean(all_accuracy)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)

    print(f'Average Accuracy: {avg_accuracy:.4f}, Average Recall: {avg_recall:.4f}, Average F1 Score: {avg_f1:.4f}')

    # 输出事件信息
    events_df = pd.DataFrame(all_events_info, columns=['Event Start', 'Event End', 'Event Position'])
    events_df.to_csv('detected_events.csv', index=False)
    print("Detected events saved to 'detected_events.csv'")

    plot_convergence_curve(convergence_curve)


if __name__ == '__main__':
    train_dir = '/mnt/data/train'  # 替换为实际训练数据文件夹路径
    test_dir = '/mnt/data/test'  # 替换为实际测试数据文件夹路径

    train_and_evaluate(train_dir, test_dir, fitness_alpha=0.5, max_epochs=100, pop_size=20)
