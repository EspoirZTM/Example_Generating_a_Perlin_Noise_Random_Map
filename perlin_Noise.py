import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def perlin_noise_2d(shape, res):
    """生成二维柏林噪声的函数"""
    def f(t):
        # 插值函数,五次方插值
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    # 创建网格
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # grid的shape为(res[0], res[1], 2)，分别表示x,y坐标和随机值
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[:-1, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # g00,g10,g01,g11分别表示四个角的梯度值
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)
    # n00,n10,n01,n11分别表示四个角的插值结果
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

def generate_terrain(map_size, mountain_range, res=(5, 5)):
    """生成地形的函数"""
    noise = perlin_noise_2d(map_size, res)
    # Normalize and scale the noise
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    noise = noise * mountain_range[1] + mountain_range[0]
    return noise

def plot_terrain(terrain):
    """绘制地形图的函数"""
    colors = [(0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
    cmap_name = 'terrain_map'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    plt.figure(figsize=(8, 6))
    plt.imshow(terrain, cmap=cm, extent=[0, terrain.shape[1], 0, terrain.shape[0]])
    plt.colorbar(label='Elevation')
    plt.title('Terrain Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 测试案例
map_size = (1000, 1000)  # Map size (rows, columns)
mountain_range = (-11034, 8848)  # of elevations (min, max)
terrain = generate_terrain(map_size, mountain_range)
plot_terrain(terrain)
