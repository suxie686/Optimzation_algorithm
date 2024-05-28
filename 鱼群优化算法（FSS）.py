import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class FishSchoolSearch:
    def __init__(self, func, n_fish=50, n_dim=2, max_iter=1000, visual=0.1, step_size=1.5):
        self.func = func  # 目标函数
        self.n_fish = n_fish  # 鱼的数量
        self.n_dim = n_dim  # 搜索空间的维度
        self.max_iter = max_iter  # 最大迭代次数
        self.visual = visual  # 视觉范围
        self.step_size = step_size  # 步长
        self.best_position = None  # 全局最优位置
        self.best_fitness = float('inf')  # 全局最优解的目标函数值
        self.fishes = np.random.uniform(-10, 10, (n_fish, n_dim))  # 初始化鱼的位置
        self.best_fitness_history = []

    def update(self):
        for _ in range(self.max_iter):
            for i in range(self.n_fish):
                # 计算当前鱼的位置到其他鱼的相对位置和距离
                relative_positions = self.fishes - self.fishes[i]
                distances = np.linalg.norm(relative_positions, axis=1)

                # 只考虑视觉范围内的鱼
                visible_mask = distances < self.visual
                if np.any(visible_mask):
                    # 向视觉范围内的最优鱼移动
                    best_fish_index = np.argmin(self.func(self.fishes[visible_mask]))
                    move_direction = self.fishes[visible_mask][best_fish_index] - self.fishes[i]
                    self.fishes[i] += self.step_size * move_direction / (1 + distances[visible_mask][best_fish_index])

                # 更新全局最优解
                fitness = self.func(self.fishes[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = self.fishes[i].copy()
                self.best_fitness_history.append(self.best_fitness)

    def optimize(self):
        self.update()
        return self.best_position, self.best_fitness


# 示例目标函数，比如找寻f(x) = x^2的最小值
def example_function(x):
    return np.sum(x ** 2)


def draw_result(best_fitness_history, max_iter, n_fish):
    plt.plot(range(max_iter * n_fish), best_fitness_history, label='best_fitness')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度')
    plt.title('鱼群优化算法（FSS）的最佳适应度')
    plt.legend()
    plt.show()


# 使用鱼群优化算法求解
fss = FishSchoolSearch(example_function)
best_pos, best_val = fss.optimize()
print("Best Position:", best_pos)
print("Best Value:", best_val)
draw_result(fss.best_fitness_history, fss.max_iter, fss.n_fish)
