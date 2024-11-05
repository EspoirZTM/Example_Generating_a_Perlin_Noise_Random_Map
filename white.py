import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def white_noise_2d(shape):
    """生成二维白噪声的函数"""
    return np.random.rand(*shape)

def generate_terrain_white_noise(map_size, mountain_range):
    """使用白噪声生成地形的函数"""
    noise = white_noise_2d(map_size)
    # Normalize and scale the noise
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    noise = noise * (mountain_range[1] - mountain_range[0]) + mountain_range[0]
    return noise

def plot_terrain(terrain):
    """绘制地形图的函数"""
    colors = [(0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
    cmap_name = 'terrain_map'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    plt.figure(figsize=(8, 6))
    plt.imshow(terrain, cmap=cm, extent=[0, terrain.shape[1], 0, terrain.shape[0]])
    plt.colorbar(label='Elevation')
    plt.title('White Noise Terrain Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 测试案例
map_size = (1000, 1000)  # Map size (rows, columns)
mountain_range = (-110, 884)  # Range of elevations (min, max)
terrain = generate_terrain_white_noise(map_size, mountain_range)
plot_terrain(terrain)
