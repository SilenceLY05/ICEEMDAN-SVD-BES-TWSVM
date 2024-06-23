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


class BasePBES(Optimizer):
    """
    Original version of: Bald Eagle Search (BES)
        (Novel meta-heuristic bald eagle search optimisation algorithm)
    Link:
        DOI: https://doi.org/10.1007/s10462-019-09732-5
    """

    def __init__(self, problem, epoch=10000, pop_size=200, a=10, beta=1.2, alpha=1.5, c1=2, c2=2, **kwargs):
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
        self.beta = beta
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = False
        self.X_min = problem.lb
        self.X_max = problem.ub

    def _create_polar_coordinates(self):
        """ Using numpy vector for faster computational time """
        # Generate uniformly distributed random numbers
        rand = np.random.uniform(0, 1, self.pop_size)

        # Archimedes' spiral for rho
        rho = rand * (self.X_max - self.X_min) + self.X_min
        # CDF inversion for theta
        theta = self.beta * np.arccos(2 * rand - 1)
        return rho, theta

    def _update_positions(self, rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## 0. Pre-definded
        rho, theta = self._create_polar_coordinates()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([individual[self.ID_POS] for individual in self.pop])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.g_best[self.ID_POS] + self.alpha * np.random.uniform() * (pos_mean - self.pop[idx][self.ID_POS])
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
            pos_new = pop_new[idx][self.ID_POS] + np.sin(theta[idx]) * (pop_new[idx][self.ID_POS] - pop_new[idx_rand][self.ID_POS]) + \
                      np.cos(theta[idx]) * (pop_new[idx][self.ID_POS] - pos_mean)
            pos_new = self.amend_position_faster(pos_new)
            pop_child.append([pos_new, None])
        pop_child = self.update_fitness_population(pop_child)
        pop_child = self.greedy_selection_population(pop_new, pop_child)

        ## 3. Swoop
        pos_list = np.array([individual[self.ID_POS] for individual in pop_child])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.uniform() * self.g_best[self.ID_POS] + np.sinh(theta[idx]) * (pop_child[idx][self.ID_POS] - self.c1 * pos_mean) \
                      + np.cosh(theta[idx]) * (pop_child[idx][self.ID_POS] - self.c2 * self.g_best[self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(pop_child, pop_new)

        ## Update polar angle using Eq. (20)
        for idx in range(0, self.pop_size):
            theta[idx] = self.alpha * theta[idx] + 2 * np.cos(2 * np.random.uniform() - 1)