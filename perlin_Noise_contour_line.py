import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def perlin_noise_2d(shape, res):
    """生成二维柏林噪声的函数"""
    def f(t):
        # 插值函数
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    # 创建网格
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[:-1, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)
    # Interpolation
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
    """绘制地形图和等高线的函数"""
    colors = [(0.3, 0.5, 1), (0, 0.5, 0), (1, 1, 0), (1, 0.5, 0)]
    cmap_name = 'terrain_map'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    plt.figure(figsize=(20, 10), dpi=100)

    # 绘制地形图
    img = plt.imshow(terrain, cmap=cm, extent=[0, terrain.shape[1], 0, terrain.shape[0]], origin='lower')

    # 添加等高线
    contour_levels = np.arange(mountain_range[0], mountain_range[1], 500)  # 根据需要调整等高线的级别
    contours = plt.contour(terrain, levels=contour_levels, colors='black', linewidths=0.5)

    # 添加 colorbar
    cbar = plt.colorbar(img, label='Elevation')


    # 添加等高线标签
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.0f', colors='black')

    plt.title('Terrain Map with Contours')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 测试案例
map_size = (100, 100)  # Map size (rows, columns)
mountain_range = (-1000, 8848)  # of elevations (min, max)
terrain = generate_terrain(map_size, mountain_range)
plot_terrain(terrain)
