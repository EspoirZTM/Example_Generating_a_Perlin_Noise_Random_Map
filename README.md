
## 摘要

本报告详细介绍了基于柏林噪声的随机地形生成方法。报告首先阐述了柏林噪声的背景及其在计算机图形学和游戏开发领域的重要性，指出传统噪声生成方法在模拟自然地形方面的局限性。随后，报告详细介绍了柏林噪声的原理及其实现过程，包括初始化、生成梯度向量、计算点积和平滑插值等关键步骤。

此外，报告还展示了如何利用柏林噪声和白噪声生成地形，并通过对比分析强调了柏林噪声在生成自然、连续地形方面的优势。最后，报告提供了地形可视化的实现方法，包括地形生成和地图绘制的关键代码，以及成果展示和源代码附录。

本研究为计算机图形学和游戏开发领域提供了一种高效、实用的地形生成技术。

## 1. 引言

柏林噪声（Perlin Noise）作为一种先进的噪声生成技术，其诞生背景与计算机图形学和游戏开发领域的需求密切相关。在20世纪80年代，随着计算机技术的发展，游戏和图形应用程序对随机生成的自然景观的需求日益增长。传统的噪声生成方法，如白噪声，由于其随机性和无规律性，往往导致生成的地形缺乏连续性和自然感。

为了解决这一问题，计算机图形学领域的研究者开始探索新的噪声生成方法。在此背景下，Patrick S. Perlin 提出了一种名为柏林噪声的噪声生成技术。该技术通过在空间中插入随机的梯度向量，并利用插值技术对噪声进行平滑处理，从而产生出具有自然外观的地形

柏林噪声的连续性和自然连贯性显著优于白噪声，为我们提供了一种更为优化的地形生成策略。这一技术的诞生，极大地推动了计算机图形学和游戏开发领域的发展，为随机地形生成提供了新的思路和方法。

## 2 柏林噪声原理及其实现
### 2.1 白噪音的局限性

在探索随机地形模型的构建方法时，研究者们通常依赖于噪声技术来确定地图上各个点的地形参数。在本研究中，我们选择了一种基础且广泛应用于信号处理的噪声类型——白噪声（White Noise）作为地形生成的函数。这一选择旨在评估其是否能准确模拟自然界中的地形特征。

```py
 def white_noise_2d(shape):
    "生成二维白噪声的函数"
     return np.random.rand(*shape)

 def generate_terrain_white_noise(map_size, mountain_range):
     "使用白噪声生成地形的函数"
     noise = white_noise_2d(map_size)
     # Normalize and scale the noise     noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
     noise = noise * (mountain_range[1] -    
      mountain_range[0]) + mountain_range[0]
     return noise

```

由此，我们得到了一张由白噪音生成的"随机地图"：

![image-20241115105526553](C:\Users\EspoirZTM\AppData\Roaming\Typora\typora-user-images\image-20241115105526553.png)

通过对比地图生成结果可以看出，可以发现白噪声在生成地形时存在显著的不平滑性，导致生成的地形缺乏连续性和自然感。这一发现凸显了白噪声在地形模拟中的局限性，从而促使我们寻求一种更为先进和高效的算法来克服这些缺陷。

柏林噪声（Perlin Noise）作为一种平滑的噪声生成方法，其连续性和自然连贯性显著优于白噪声，为地形生成提供了一种更为优化的策略。

### 2.2 柏林噪声的函数实现

#### 2.2.1 初始化

在柏林噪声的地形生成流程中，初始化阶段扮演着核心角色，该阶段旨在确立地形构建的基石。具体而言，该阶段涉及构建一个网格体系，并对其内部各节点的位置进行精确标定。此外，为确保地形的模块化与区域划分，该阶段亦引入了区块概念，旨在为地形生成提供有序的框架和指导。

<img src="C:\Users\EspoirZTM\AppData\Roaming\Typora\typora-user-images\image-20241115112132059.png" alt="image-20241115112132059" style="zoom:50%;" />

```py
delta = (res[0] / shape[0], res[1] / shape[1])
d = (shape[0] // res[0], shape[1] // res[1])
grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
```

#### 2.2.2 生成梯度向量

为网格中每个单元格的四个角点分别生成一组随机的梯度向量，这一步骤对于指导噪声的生成过程至关重要。在柏林噪声算法中，通过为每个单元格的角点生成一组随机的梯度向量，我们可以为地形模拟提供一种具有方向性的噪声分布。这种方向性使得地形在平滑过渡的同时，还能够展现出局部的高低差异和细节特征。

<img src="C:\Users\EspoirZTM\AppData\Roaming\Typora\typora-user-images\image-20241115112158163.png" alt="image-20241115112158163" style="zoom:50%;" />

```py
angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
gradients = np.dstack((np.cos(angles), np.sin(angles)))
g00 = gradients[:-1, :-1].repeat(d[0], 0).repeat(d[1], 1)
g10 = gradients[1:, :-1].repeat(d[0], 0).repeat(d[1], 1)
g01 = gradients[:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
```

#### 2.2.3 计算点积

在生成梯度向量之后，通过计算网格点与这些梯度向量的点积，可以得到每个点的噪声值。这一步骤实质上是将随机的梯度信息与网格点的位置信息相结合，从而形成一种连续且动态的噪声分布。

<img src="https://kenshin.tech/2023/03/01/%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%9B%BE%E5%BD%A2%E5%AD%A6%EF%BC%88%E5%8D%81%EF%BC%89%EF%BC%9A%E6%9F%8F%E6%9E%97%E5%99%AA%E5%A3%B0/1.jpg" alt="perlinNoise" style="zoom:50%;" />

```py
n00 = np.sum(grid * g00, axis=2)
n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)
```

#### 2.2.4 平滑插值

我们先前得到了四张由区块得到的地形渲染图，每个区块的点积值都经过了颜色映射，反映出当前位置的海拔。但这样的地形在区块边界处呈现了明显的不连续现象，因此需要对这些边界值进行平滑处理。

柏林噪声算法采用了五次方插值函数对噪声进行处理。这种插值函数能够有效地减少噪声中的高频成分，从而生成更为平滑和连续的地形。

柏林噪声一般采用采用多项式插值。按照特定系数的多项式对每个区块进行平滑处理
$$
6  t^5 - 15  t^4 + 10  t^3
$$
![](C:\Users\EspoirZTM\Desktop\图片2.png)

```python
t = f(grid)
n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
```

## 3. 地图可视化的实现

### 3.1 地形生成generate_terrain函数

#### 3.1.1 函数定义

为了将噪声映射到地图上，现定义一个函数：

```python
def (map_size, mountain_range, res=(5, 5)):
```

该函数接收三个参数：

`map_size`：表示生成地形的大小，以元组形式给出（行数, 列数）。

`mountain_range`：表示地形高度的范围，以元组形式给出（最小高度, 最大高度）。

`res`：噪声分辨率的元组，默认值为 (5, 5)。

#### 3.1.2 生成柏林噪声

```python
noise = perlin_noise_2d(map_size, res)
```

该语句调用 `perlin_noise_2d` 函数生成二维柏林噪声，返回的噪声数组赋值给 `noise`。这个数组将作为基础的地形特征图。

#### 3.1.3 归一化和缩放噪声

```python
noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
```

这一行代码的作用是将`noise`数组中的值归一化到 [0, 1] 的范围内，其中：

`np.min(noise)` 计算 `noise` 数组中的最小值。

`np.max(noise)` 计算 `noise` 数组中的最大值。

通过减去最小值并除以最大值和最小值之间的差，所有噪声值都被转换到 0 到 1 的区间。

#### 3.1.4 调整噪声值到地形高度范围

```python
noise = noise * mountain_range[1] + mountain_range[0]
```

此行代码将归一化后的噪声值重新调整到指定的地形高度范围。其中：

`mountain_range[1]` 是最高高度。

`mountain_range[0]` 是最低高度。

通过乘以最大高度和加上最小高度，所有的噪声值被调整到给定的海拔范围内。

### 3.2 地图绘制plot_terrain函数
本研究将采用`plot_terrain`函数以生成地形图及相应的等高线图。该函数位于perlin_Noise_contour_line.py脚本文件中。以下为执行地形图及等高线图绘制的关键步骤：

#### 3.2.1 函数定义

```python
def plot_terrain(terrain):
```



这里定义了一个函数 `plot_terrain`，它接受一个参数 `terrain`，这个参数应该是一个二维数组，表示地形的高程值。

#### 3.2.2 颜色定义

```python
colors = [(0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0)]
```

​	该代码段定义了一个颜色集，采用元组格式表示RGBA（红、绿、蓝、透明度）值。具体而言，所列颜色依次为：

- 淡蓝色（0, 0.5, 1）

- 绿色（0, 1, 0）
- 黄色（1, 1, 0
- 橙色（1, 0.5, 0）

#### 3.2.3 创建颜色映射

```python
cmap_name = 'terrain_map'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
```

- `cmap_name` 定义了颜色映射的名称。
- `LinearSegmentedColormap.from_list` 创建了一个线性分段颜色映射（colormap），通过给定的颜色列表生成，`N=100` 表示在这个颜色映射中有100个不同的颜色值。

#### 3.2.4 创建图形

```python
plt.figure(figsize=(8, 6))
```

这行代码利用 `plt.figure` 创建了一个新的图形窗口，其大小为 8x6 英寸。

#### 3.2.5 绘制地形图像

```python
plt.imshow(terrain, cmap=cm, extent=[0, terrain.shape[1], 0, terrain.shape[0]])
```

- `plt.imshow` 用于显示图像数据，这里传入的 `terrain` 是要展示的高程数据。
- `cmap=cm` 设定使用之前创建的颜色映射。
- `extent` 参数设定了坐标轴的范围，这里根据 `terrain` 的形状决定了x轴和y轴的范围。

#### 3.2.6 添加颜色条

```python
plt.colorbar(label='Elevation')
```

这行代码添加了一个颜色条，以表示不同颜色对应的高程值，并用“Elevation”作为标签。

#### 3.2.7 设置标题和标签

```python
plt.title('White Noise Terrain Map')
plt.xlabel('X')
plt.ylabel('Y')
```

`plt.title` 为图形设置标题。

`plt.xlabel`   和 `plt.ylabel` 为x轴和y轴分别添加标签。

#### 3.2.8 显示图形

```python
plt.show()
```

这行代码用来最终展示绘制的图形。

## 5. 成果展示

![](C:\Users\EspoirZTM\Desktop\PNG\Figure_5.png)

## 6、总结与展望

### 6.1总结

本报告探讨了基于柏林噪声的随机地形生成方法。首先，我们讨论了柏林噪声在计算机图形学和游戏开发领域的应用背景，并指出其相较于传统白噪声在生成自然、连续地形方面的优势。接着，报告详细介绍了柏林噪声的原理和实现步骤，包括初始化、生成梯度向量、计算点积和平滑插值等关键步骤。此外，我们还展示了如何利用柏林噪声和白噪声生成地形，并通过对比分析强调了柏林噪声在地形模拟中的优势。

### 6.2 展望

#### 6.2.1 **柏林噪声的改进方案**

探索深度学习技术的集成，通过神经网络对柏林噪声算法进行优化，旨在提升地形生成的质量和效率。同时，研究柏林噪声与其他噪声生成算法的融合，寻求创新性的混合噪声生成策略，以进一步增强地形模拟的真实性。

#### 6.2.2 **深度学习与地形生成的结合**

通过深度学习模型的自动化特征提取，我们能够实现更为精细和多样化的地形生成。结合生成对抗网络（GAN）等先进技术，能够进一步实现更为真实和随机的地形生成。此外，我们还致力于探索深度学习在实时地形生成中的应用，旨在为游戏和虚拟现实等领域提供更为流畅和逼真的用户体验。

#### 6.2.3 **跨学科研究**

结合地理信息系统（GI）和遥感技术，实现对真实地形的逼真模拟和分析。

研究柏林噪声在地形演变、地质模拟等领域的应用，为相关学科研究提供有力支持。

## 附录：源代码呈现

### 柏林噪声源码

```python
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
```

### 白噪声源码

```py
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
```

## 参考文献

> Ken Perlin. 1985. An image synthesizer. SIGGRAPH Comput. Graph. 19, 3 (Jul. 1985), 287–296. https://doi.org/10.1145/325165.325247
