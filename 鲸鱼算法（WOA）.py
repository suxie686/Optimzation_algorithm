import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class WhaleOptimizationAlgorithm:
    def __init__(self, func, dim, n_pop, max_iter, lb, ub):
        self.func = func  # 目标函数
        self.dim = dim  # 搜索空间维度
        self.n_pop = n_pop  # 粒子群体数量
        self.max_iter = max_iter  # 最大迭代次数
        self.lb = lb  # 搜索空间下界
        self.ub = ub  # 搜索空间上界
        self.pop = np.random.uniform(lb, ub, (n_pop, dim))  # 初始化种群
        self.best_solution = self.pop[np.argmin([self.func(x) for x in self.pop])]  # 初始化最优解
        self.best_solution_fitness = self.func(self.best_solution)
        self.best_solution_fitness_history = []

    def update_position(self, a, c, l):
        """
        更新位置
        :param a: 缩放因子
        :param c: 常数
        :param l: 迭代计数
        """
        d = np.abs(c * self.best_solution - self.pop)
        new_position = self.best_solution - a * d
        return np.clip(new_position, self.lb, self.ub)

    def search(self):
        for iter in range(self.max_iter):
            a = 2 * np.exp(-4 * iter / self.max_iter)  # 缩放因子随迭代减少
            c = 2 * np.random.rand() - 1  # 随机漫步因子

            for i in range(self.n_pop):
                l = np.random.rand()  # 计算每个鲸鱼的位置更新水平

                if l < 0.5:
                    self.pop = self.update_position(a, c, l)
                elif l >= 0.5 and l < 0.9:
                    random_index = np.random.randint(0, self.n_pop)
                    self.pop[i] = self.pop[random_index] - a * np.abs(
                        np.random.rand() * (self.pop[random_index] - self.pop[i]))
                else:
                    self.pop[i] = self.best_solution + 0.001 * np.random.randn(self.dim)

                # 确保新位置在边界内
                self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)

                # 更新最优解
                if self.func(self.pop[i]) < self.func(self.best_solution):
                    self.best_solution = self.pop[i].copy()

            print(f"Iteration {iter + 1}, Best Fitness: {self.func(self.best_solution)}")
            self.best_solution_fitness_history.append(self.func(self.best_solution))

        return self.best_solution, self.func(self.best_solution)


# 示例目标函数，比如寻找最小值的函数
def sphere_func(x):
    return sum(xi ** 2 for xi in x)


def draw_result(best_solution_fitness_history, len):
    plt.plot(range(len), best_solution_fitness_history, label='best_fitness')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度')
    plt.title('鱼群优化算法（FSS）的最佳适应度')
    plt.legend()
    plt.show()


# 使用示例
woa = WhaleOptimizationAlgorithm(func=sphere_func, dim=10, n_pop=30, max_iter=100, lb=-10, ub=10)
best_solution, best_fitness = woa.search()
draw_result(woa.best_solution_fitness_history, len(woa.best_solution_fitness_history))
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
