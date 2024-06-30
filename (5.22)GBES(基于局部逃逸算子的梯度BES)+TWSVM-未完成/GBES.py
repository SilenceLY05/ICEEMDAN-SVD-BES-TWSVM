#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGBES(Optimizer):
    """
    Original version of: Bald Eagle Search (BES)
        (Novel meta-heuristic bald eagle search optimisation algorithm)
    Link:
        DOI: https://doi.org/10.1007/s10462-019-09732-5
    """

    def __init__(self, problem, epoch=500, pop_size=100, a=10, R=1.5, alpha=1.5, c1=2, c2=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a (int): default: 10, determining the corner between point search in the central point, in [5, 10]
            R (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
            alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
            c1 (float): default: 2, in [1, 2]
            c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.a = a
        self.R = R
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = False
        self.X_min = problem.lb
        self.X_max = problem.ub

    def _create_x_y_x1_y1_(self):
        """ Using numpy vector for faster computational time """
        ## Eq. 2
        phi = self.a * np.pi * np.random.uniform(0, 1, self.pop_size)
        r = phi + self.R * np.random.uniform(0, 1, self.pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)

        ## Eq. 3
        r1 = phi1 = self.a * np.pi * np.random.uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)

        x_list = xr / max(xr)
        y_list = yr / max(yr)
        x1_list = xr1 / max(xr1)
        y1_list = yr1 / max(yr1)
        return x_list, y_list, x1_list, y1_list

    def _compute_rho1(self, current_iter):
        """
        根据当前迭代次数和总迭代次数计算 rho1
        """
        b = 0.2 + (1 - (current_iter / self.epoch) ** 3) ** 2
        alpha = abs(b * np.sin(3 * np.pi / 2 + np.sin(b * 3 * np.pi / 2)))
        rho1 = 2 * np.random.rand() * alpha - alpha
        return rho1

    def _local_escaping_operator(self, x_best, x, x_k, X1_n, X2_n, X_r1, X_r2, pr, rho1):
        """
        应用局部逃逸算子 (LEO) 生成新解
        """
        f1 = np.random.uniform(-1, 1)
        f2 = np.random.normal(0, 1)
        rand_val = np.random.rand()
        u1 = 2 * rand_val if np.random.rand() < 0.5 else 1
        u2 = rand_val if np.random.rand() < 0.5 else 1
        u3 = rand_val if np.random.rand() < 0.5 else 1
        if np.random.rand() < pr:
            if np.random.rand() < 0.5:
                X_LEO = x + f1 * (u1 * x_best - u2 * x_k) + f2 * rho1 * (u3 * (X2_n - X1_n) + u2 * (X_r1 - X_r2)) / 2
                x=X_LEO
            else:
                X_LEO = x_best + f1 * (u1 * x_best - u2 * x_k) + f2 * rho1 * (u3 * (X2_n - X1_n) + u2 * (X_r1 - X_r2)) / 2
                x = X_LEO
        return x

    def _compute_x_k(self, pos_list):
        """
        根据当前种群位置计算新的候选解 x_k
        """
        l2 = np.random.rand()
        if l2 < 0.5:
            x_rand = self.X_min + np.random.rand() * (self.X_max - self.X_min)
        else:
            x_rand = pos_list[np.random.randint(0, len(pos_list))]
        return x_rand

    def _find_similar_and_dissimilar(self, current_pos, pos_list):
        """
        找到与当前个体位置最相似和最不相似的解
        """
        distances = np.linalg.norm(pos_list - current_pos, axis=1)
        closest_idx = np.argmin(distances + np.eye(len(distances)) * 1e10)  # 避免选择自身
        furthest_idx = np.argmax(distances)
        return pos_list[closest_idx], pos_list[furthest_idx]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## 0. Pre-definded
        x_list, y_list, x1_list, y1_list = self._create_x_y_x1_y1_()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([individual[self.ID_POS] for individual in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        alpha2 = self.alpha * (self.epoch - epoch +1) / self.epoch

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.g_best[self.ID_POS] + alpha2 * np.random.uniform() * (pos_mean - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)

        ## 2. Search in space
        pos_list = np.array([individual[self.ID_POS] for individual in pop_new])
        pos_mean = np.mean(pos_list, axis=0)

        pop_child = []
        for idx in range(0, self.pop_size):
            idx_rand = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos_new = pop_new[idx][self.ID_POS] + y_list[idx] * (pop_new[idx][self.ID_POS] - pop_new[idx_rand][self.ID_POS]) + \
                      x_list[idx] * (pop_new[idx][self.ID_POS] - pos_mean)
            pos_new = self.amend_position_faster(pos_new)
            pop_child.append([pos_new, None])
        pop_child = self.update_fitness_population(pop_child)
        pop_child = self.greedy_selection_population(pop_new, pop_child)

        ## 3. Swoop
        pos_list = np.array([individual[self.ID_POS] for individual in pop_child])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.uniform() * self.g_best[self.ID_POS] + x1_list[idx] * (pop_child[idx][self.ID_POS] - self.c1 * pos_mean) \
                      + y1_list[idx] * (pop_child[idx][self.ID_POS] - self.c2 * self.g_best[self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(pop_child, pop_new)

        ## 4. Apply Local Escaping Operator (LEO)
        pr = 0.5  # 设置 pr 值
        for idx in range(0, self.pop_size):
            X1_n, X2_n = self._find_similar_and_dissimilar(pop_new[idx][self.ID_POS], pos_list)
            X_r1 = pos_list[np.random.randint(0, self.pop_size)]
            X_r2 = pos_list[np.random.randint(0, self.pop_size)]
            x_k = self._compute_x_k(pos_list)
            rho1 = self._compute_rho1(epoch)
            pos_new = self._local_escaping_operator(self.g_best[self.ID_POS], pop_new[idx][self.ID_POS],
                                                    x_k, X1_n, X2_n, X_r1, X_r2, pr, rho1)
            pos_new = self.amend_position_faster(pos_new)
            pop_new[idx] = [pos_new, None]

        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(pop_child, pop_new)
