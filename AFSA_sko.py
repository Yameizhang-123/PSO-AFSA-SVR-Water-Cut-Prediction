from sko.AFSA import AFSA
import numpy as np
from scipy import spatial
from tqdm import tqdm

# overrided class AFSA
class AFSA_sko(AFSA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=300,
                 max_try_num=100, step=0.5, visual=0.3,
                 q=0.98, delta=0.5, X=None):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num  # 最大尝试捕食次数
        self.step = step  # 每一步的最大位移比例
        self.visual = visual  # 鱼的最大感知范围
        self.q = q  # 鱼的感知范围衰减系数
        self.delta = delta  # 拥挤度阈值，越大越容易聚群和追尾

        # self.X = np.random.rand(self.size_pop, self.n_dim)
        self.X = np.array([[np.random.normal(loc=X[0], scale=X[0] * 0.2, size=self.size_pop)],
                           [np.random.normal(loc=X[1], scale=X[1] * 0.2, size=self.size_pop)],
                           [np.random.normal(loc=X[2], scale=X[2] * 0.2, size=self.size_pop)]
                           ]).reshape(n_dim, size_pop).transpose(1, 0)
        self.X = self.X.clip(0, np.inf)
        self.Y = np.array([self.func(x) for x in self.X])

        best_idx = self.Y.argmin()
        self.best_x, self.best_y = self.X[best_idx, :], self.Y[best_idx]
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase

    def move_to_target(self, idx_individual, x_target):
        '''
        move to target
        called by prey(), swarm(), follow()

        :param idx_individual:
        :param x_target:
        :return:
        '''
        x = self.X[idx_individual, :]
        x_new = x + self.step * np.random.rand() * (x_target - x)
        # x_new = x_target
        x_new = np.clip(x_new, 0.000000001, np.inf)
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_x = self.X[idx_individual, :].copy()
            self.best_y = self.Y[idx_individual].copy()

    def move(self, idx_individual):
        '''
        randomly move to a point

        :param idx_individual:
        :return:
        '''
        r = 2 * np.random.rand(self.n_dim) - 1
        x_new = self.X[idx_individual, :] + self.visual * r
        x_new = np.clip(x_new, 0.000000001, np.inf)
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_X = self.X[idx_individual, :].copy()
            self.best_Y = self.Y[idx_individual].copy()

    def prey(self, idx_individual):
        '''
        prey
        :param idx_individual:
        :return:
        '''
        for try_num in range(self.max_try_num):
            r = 2 * np.random.rand(self.n_dim) - 1
            x_target = self.X[idx_individual, :] + self.visual * r
            x_target = np.clip(x_target, 0.000000001, np.inf)
            if self.func(x_target) < self.Y[idx_individual]:  # 捕食成功
                self.move_to_target(idx_individual, x_target)
                return None
        # 捕食 max_try_num 次后仍不成功，就调用 move 算子
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        # 找出 idx_individual 这条鱼视线范围内的所有鱼
        distances = spatial.distance.cdist(self.X[[idx_individual], :], self.X, metric='euclidean').reshape(-1)
        idx_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]
        return idx_individual_in_vision

    def swarm(self, idx_individual):
        # 聚群行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            center_individual_in_vision = individual_in_vision.mean(axis=0)
            center_y_in_vision = self.func(center_individual_in_vision)
            if center_y_in_vision * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center_individual_in_vision)
                return None
        self.prey(idx_individual)

    def follow(self, idx_individual):
        # 追尾行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            y_in_vision = np.array([self.func(x) for x in individual_in_vision])
            idx_target = y_in_vision.argmin()
            x_target = individual_in_vision[idx_target]
            y_target = y_in_vision[idx_target]
            if y_target * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return None
        self.prey(idx_individual)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        # #不显示进度条用121-126
        # for epoch in range(self.max_iter):
        #     for idx_individual in range(self.size_pop):
        #         self.swarm(idx_individual)
        #         self.follow(idx_individual)
        #     self.visual *= self.q
        # self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase

        # 显示进度条129-135
        for epoch in tqdm(range(self.max_iter)):
            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.follow(idx_individual)
            self.visual *= self.q
            print('epoch: {}, best_x: {}, best_y: {}'.format(epoch, self.best_x, self.best_y))
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase
        return self.best_x, self.best_y
