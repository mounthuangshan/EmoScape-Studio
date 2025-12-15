import numpy as np
from scipy.ndimage import gaussian_filter

def identify_valleys(height_map, threshold=0.6):
    """识别山谷区域
    
    Args:
        height_map: 高度图
        threshold: 鉴别阈值
        
    Returns:
        valleys: 山谷位置布尔数组
    """
    # 计算高度的拉普拉斯算子(二阶导数)
    from scipy.ndimage import laplace
    
    # 计算拉普拉斯
    lap = laplace(gaussian_filter(height_map, sigma=1.0))
    
    # 正值表示山谷(凹形)，负值表示山脊(凸形)
    valleys = lap > threshold
    
    # 过滤掉太小的区域
    from scipy.ndimage import binary_opening
    valleys = binary_opening(valleys, structure=np.ones((3, 3)))
    
    return valleys

def identify_plateaus(height_map, flatness_threshold=0.05, height_percentile=70):
    """识别高原区域
    
    Args:
        height_map: 高度图
        flatness_threshold: 平坦度阈值(坡度小于此值被视为平坦)
        height_percentile: 高度百分位(超过此值的高度被视为高地)
        
    Returns:
        plateaus: 高原位置布尔数组
    """
    # 计算梯度(坡度)
    grad_y, grad_x = np.gradient(height_map)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 找到平坦区域
    flat_areas = gradient_magnitude < flatness_threshold
    
    # 找到高地区域(高于height_percentile)
    height_threshold = np.percentile(height_map, height_percentile)
    high_areas = height_map > height_threshold
    
    # 高原是平坦且高的区域
    plateaus = flat_areas & high_areas
    
    # 过滤掉太小的区域
    from scipy.ndimage import binary_opening
    plateaus = binary_opening(plateaus, structure=np.ones((3, 3)))
    
    return plateaus

def identify_coastal_areas(height_map, water_threshold=0.2, distance_threshold=10):
    """识别沿海区域
    
    Args:
        height_map: 高度图
        water_threshold: 水体高度阈值
        distance_threshold: 离海岸最大距离
        
    Returns:
        coastal_areas: 沿海区域布尔数组
    """
    # 找到水域
    water_bodies = height_map < water_threshold
    
    # 找到陆地
    land = ~water_bodies
    
    # 计算到海岸的距离
    from scipy.ndimage import distance_transform_edt
    
    # 陆地上到水域的距离
    dist_to_water = distance_transform_edt(land)
    
    # 水域上到陆地的距离
    dist_to_land = distance_transform_edt(water_bodies)
    
    # 沿海区域是距离海岸线较近的区域
    coastal_areas = (dist_to_water < distance_threshold) & land
    
    return coastal_areas

def calculate_valley_climate(temp, humid, height_map, x, y, radius=5):
    """计算山谷微气候特性
    
    Args:
        temp: 基础温度值
        humid: 基础湿度值
        height_map: 高度图
        x, y: 位置坐标
        radius: 考虑范围半径
        
    Returns:
        microclimate_value: 微气候影响值(用于制作微气候图的索引)
    """
    height, width = height_map.shape
    
    # 提取局部区域
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    
    local_heights = height_map[y_min:y_max, x_min:x_max]
    
    # 计算局部高度统计
    current_height = height_map[y, x]
    mean_height = np.mean(local_heights)
    
    # 山谷通常比周围区域略冷(温度降低)
    temp_mod = -0.05 if current_height < mean_height else 0
    
    # 山谷通常比周围区域湿度高
    humid_mod = 0.1 if current_height < mean_height else 0
    
    # 生成微气候影响值(可以根据需要扩展)
    microclimate_value = 1  # 1表示山谷微气候
    
    return microclimate_value

def calculate_plateau_climate(temp, humid, height_map, x, y, radius=5):
    """计算高原微气候特性
    
    Args:
        temp: 基础温度值
        humid: 基础湿度值
        height_map: 高度图
        x, y: 位置坐标
        radius: 考虑范围半径
        
    Returns:
        microclimate_value: 微气候影响值
    """
    height, width = height_map.shape
    
    # 提取局部区域
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    
    local_heights = height_map[y_min:y_max, x_min:x_max]
    
    # 计算局部高度差异
    current_height = height_map[y, x]
    height_std = np.std(local_heights)
    
    # 高原通常温差大(这里简化为整体温度略低)
    temp_mod = -0.1 if height_std < 0.05 else 0
    
    # 高原通常湿度低
    humid_mod = -0.15 if height_std < 0.05 else 0
    
    # 生成微气候影响值
    microclimate_value = 2  # 2表示高原微气候
    
    return microclimate_value

def calculate_coastal_climate(temp, humid, height_map, x, y, radius=5):
    """计算沿海微气候特性
    
    Args:
        temp: 基础温度值
        humid: 基础湿度值
        height_map: 高度图
        x, y: 位置坐标
        radius: 考虑范围半径
        
    Returns:
        microclimate_value: 微气候影响值
    """
    # 沿海区域温度受海洋调节(较为温和)
    temp_mod = 0
    
    # 沿海区域湿度高
    humid_mod = 0.2
    
    # 生成微气候影响值
    microclimate_value = 3  # 3表示沿海微气候
    
    return microclimate_value

def generate_microclimates(height_map, temp_map, humid_map):
    """生成基于地形和主气候的微气候区域
    
    Args:
        height_map: 高度图
        temp_map: 温度图
        humid_map: 湿度图
        
    Returns:
        microclimate_map: 微气候类型图(整数索引表示不同微气候)
        adjusted_temp_map: 根据微气候调整后的温度图
        adjusted_humid_map: 根据微气候调整后的湿度图
    """
    height, width = height_map.shape
    microclimate_map = np.zeros((height, width), dtype=np.int32)
    
    # 复制输入图以防修改原始数据
    adjusted_temp = temp_map.copy()
    adjusted_humid = humid_map.copy()
    
    # 识别特殊地形特征
    valleys = identify_valleys(height_map)
    plateaus = identify_plateaus(height_map)
    coastal_areas = identify_coastal_areas(height_map)
    
    # 为每个区域分配微气候特性
    for y in range(height):
        for x in range(width):
            if valleys[y, x]:
                # 山谷微气候：温度适中，湿度较高
                microclimate_map[y, x] = calculate_valley_climate(
                    temp_map[y, x], humid_map[y, x], height_map, x, y)
                # 调整温度和湿度
                adjusted_temp[y, x] = temp_map[y, x] - 0.05
                adjusted_humid[y, x] = min(1.0, humid_map[y, x] + 0.1)
            elif plateaus[y, x]:
                # 高原微气候：昼夜温差大，湿度低
                microclimate_map[y, x] = calculate_plateau_climate(
                    temp_map[y, x], humid_map[y, x], height_map, x, y)
                # 调整温度和湿度
                adjusted_temp[y, x] = temp_map[y, x] - 0.1
                adjusted_humid[y, x] = max(0.0, humid_map[y, x] - 0.15)
            elif coastal_areas[y, x]:
                # 沿海微气候：温度受海洋调节，湿度高
                microclimate_map[y, x] = calculate_coastal_climate(
                    temp_map[y, x], humid_map[y, x], height_map, x, y)
                # 调整温度和湿度
                adjusted_temp[y, x] = temp_map[y, x] * 0.9 + 0.1  # 更温和
                adjusted_humid[y, x] = min(1.0, humid_map[y, x] + 0.2)
    
    # 确保调整后的值在有效范围内
    adjusted_temp = np.clip(adjusted_temp, 0, 1)
    adjusted_humid = np.clip(adjusted_humid, 0, 1)
    
    # 对调整后的地图进行平滑，避免硬边界
    adjusted_temp = gaussian_filter(adjusted_temp, sigma=1.0)
    adjusted_humid = gaussian_filter(adjusted_humid, sigma=1.0)
    
    return microclimate_map, adjusted_temp, adjusted_humid