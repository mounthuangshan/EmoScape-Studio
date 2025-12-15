import numpy as np
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_mountain_shadows(height_map, sun_angle=(1.0, 1.0)):
    """计算山体阴影效应
    
    Args:
        height_map: 高度图数组
        sun_angle: 太阳角度向量(x,y)
        
    Returns:
        shadow_map: 阴影图(0-1)，0表示无阴影，1表示完全阴影
    """
    height, width = height_map.shape
    shadow_map = np.zeros_like(height_map)
    
    # 归一化太阳方向
    sun_x, sun_y = sun_angle
    magnitude = np.sqrt(sun_x**2 + sun_y**2)
    if magnitude > 0:
        sun_x, sun_y = sun_x / magnitude, sun_y / magnitude
    else:
        sun_x, sun_y = 1.0, 0.0  # 默认太阳从东方升起
    
    # 计算梯度(坡度)
    grad_y, grad_x = np.gradient(height_map)
    
    # 计算表面法线与太阳方向的点积
    # 法线是(-grad_x, -grad_y, 1)，归一化后的
    norm = np.sqrt(grad_x**2 + grad_y**2 + 1)
    normal_x = -grad_x / norm
    normal_y = -grad_y / norm
    normal_z = 1.0 / norm
    
    # 太阳方向向量(sun_x, sun_y, sun_z)，其中sun_z是太阳高度
    sun_z = 0.5  # 太阳高度，可调整
    
    # 计算阴影强度(点积)
    shadow_strength = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z
    
    # 转换为阴影图(将负值变为阴影，正值为光照)
    shadow_map = np.clip(1.0 - shadow_strength, 0, 1)
    
    # 模拟阴影投射
    shadow_map = cast_shadows(height_map, shadow_map, sun_angle)
    
    return shadow_map

def cast_shadows(height_map, shadow_map, sun_angle, max_distance=30):
    """模拟阴影投射效果
    
    Args:
        height_map: 高度图
        shadow_map: 初始阴影图
        sun_angle: 太阳角度向量
        max_distance: 最大投射距离
        
    Returns:
        cast_shadow_map: 包含投射阴影的阴影图
    """
    height, width = height_map.shape
    cast_shadow_map = shadow_map.copy()
    
    # 归一化太阳方向
    sun_x, sun_y = sun_angle
    magnitude = np.sqrt(sun_x**2 + sun_y**2)
    if magnitude > 0:
        sun_x, sun_y = sun_x / magnitude, sun_y / magnitude
    else:
        return shadow_map  # 无方向，无阴影投射
    
    # 计算太阳高度系数
    sun_z = 0.5  # 太阳高度，可调整
    
    # 对每个高点投射阴影
    for y in range(height):
        for x in range(width):
            current_height = height_map[y, x]
            
            # 沿太阳相反方向投射
            for distance in range(1, max_distance):
                shadow_x = int(x - sun_x * distance)
                shadow_y = int(y - sun_y * distance)
                
                # 检查边界
                if 0 <= shadow_x < width and 0 <= shadow_y < height:
                    # 计算高度差与距离关系
                    height_diff = current_height - height_map[shadow_y, shadow_x]
                    if height_diff > 0:
                        # 如果当前点高于投射点，形成阴影
                        shadow_intensity = min(1.0, height_diff / (distance * sun_z))
                        cast_shadow_map[shadow_y, shadow_x] = max(
                            cast_shadow_map[shadow_y, shadow_x], 
                            shadow_intensity
                        )
    
    # 平滑阴影边缘
    cast_shadow_map = gaussian_filter(cast_shadow_map, sigma=1.0)
    
    return cast_shadow_map

def apply_shadow_effect(temp_map, shadow_map, shadow_temp_effect=5.0):
    """根据阴影图调整温度
    
    Args:
        temp_map: 温度图
        shadow_map: 阴影图
        shadow_temp_effect: 阴影导致的最大温度变化
        
    Returns:
        adjusted_temp_map: 调整后的温度图
    """
    adjusted_temp = temp_map.copy()
    
    # 阴影区域降温，阳光区域升温
    temp_adjustment = (0.5 - shadow_map) * shadow_temp_effect
    adjusted_temp += temp_adjustment
    
    # 确保温度范围合理(假设0-1范围)
    adjusted_temp = np.clip(adjusted_temp, 0, 1)
    
    return adjusted_temp

def identify_water_bodies(height_map, water_threshold=0.2):
    """识别水体区域
    
    Args:
        height_map: 高度图
        water_threshold: 水体高度阈值
        
    Returns:
        water_bodies: 水体位置布尔数组
    """
    return height_map < water_threshold

def update_local_humidity(humid_map, water_bodies, influence_radius=5, influence_strength=0.3):
    """更新水体周围的湿度
    
    Args:
        humid_map: 湿度图
        water_bodies: 水体位置布尔数组
        influence_radius: 影响半径
        influence_strength: 影响强度
        
    Returns:
        updated_humid_map: 更新后的湿度图
    """
    height, width = humid_map.shape
    updated_humid = humid_map.copy()
    
    # 对每个水体单元格，增加周围的湿度
    y_indices, x_indices = np.where(water_bodies)
    
    for y, x in zip(y_indices, x_indices):
        # 确定影响范围
        y_min = max(0, y - influence_radius)
        y_max = min(height, y + influence_radius + 1)
        x_min = max(0, x - influence_radius)
        x_max = min(width, x + influence_radius + 1)
        
        # 计算距离和影响强度
        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                distance = np.sqrt((ny - y)**2 + (nx - x)**2)
                if distance <= influence_radius:
                    # 影响强度随距离衰减
                    effect = influence_strength * (1 - distance / influence_radius)
                    updated_humid[ny, nx] += effect * (1 - updated_humid[ny, nx])
    
    # 确保湿度范围在0-1之间
    updated_humid = np.clip(updated_humid, 0, 1)
    
    return updated_humid

def calculate_orographic_precipitation(height_map, humid_map, wind_direction=(1.0, 0.0)):
    """计算地形性降水
    
    Args:
        height_map: 高度图
        humid_map: 湿度图
        wind_direction: 风向向量(x,y)
        
    Returns:
        precipitation: 降水量数组
    """
    height, width = height_map.shape
    precipitation = np.zeros_like(height_map)
    
    # 归一化风向
    wind_x, wind_y = wind_direction
    magnitude = np.sqrt(wind_x**2 + wind_y**2)
    if magnitude > 0:
        wind_x, wind_y = wind_x / magnitude, wind_y / magnitude
    else:
        wind_x, wind_y = 1.0, 0.0  # 默认风从东向西
    
    # 计算高度梯度
    grad_y, grad_x = np.gradient(height_map)
    
    # 风向与梯度的点积表示上坡或下坡效应
    upslope = grad_x * wind_x + grad_y * wind_y
    
    # 正值表示上坡(增加降水)，负值表示下坡(减少降水)
    # 将上坡效应与湿度相乘
    precipitation = np.clip(upslope, 0, None) * humid_map
    
    # 平滑降水分布
    precipitation = gaussian_filter(precipitation, sigma=1.5)
    
    return precipitation

def update_humidity_from_precipitation(humid_map, precipitation, precip_factor=0.3, recovery_factor=0.1):
    """根据降水更新湿度
    
    Args:
        humid_map: 湿度图
        precipitation: 降水量数组
        precip_factor: 降水消耗湿度的系数
        recovery_factor: 湿度恢复系数(小雨后的蒸发)
        
    Returns:
        updated_humid_map: 更新后的湿度图
    """
    updated_humid = humid_map.copy()
    
    # 降水消耗湿度
    humidity_loss = precipitation * precip_factor
    updated_humid -= humidity_loss
    
    # 轻微降水后的湿度恢复(如地面蒸发)
    small_precip_mask = (0 < precipitation) & (precipitation < 0.2)
    updated_humid[small_precip_mask] += recovery_factor
    
    # 确保湿度范围在0-1之间
    updated_humid = np.clip(updated_humid, 0, 1)
    
    return updated_humid

def simulate_ecosystem_dynamics(height_map, temp_map, humid_map, iterations=3):
    """模拟生态系统动态演化，整合多个环境因素的相互作用
    
    Args:
        height_map: 高度图
        temp_map: 温度图
        humid_map: 湿度图
        iterations: 迭代次数，越多越精细但计算成本更高
        
    Returns:
        enhanced_temp_map: 增强后的温度图
        enhanced_humid_map: 增强后的湿度图
    """
    # 复制初始图以防修改原始数据
    current_temp = temp_map.copy()
    current_humid = humid_map.copy()
    
    for i in range(iterations):
        # 太阳角度随迭代变化，模拟一天中的太阳位置
        sun_angle = (np.cos(i/iterations * np.pi * 2), np.sin(i/iterations * np.pi * 2))
        
        # 计算山体阴影效应
        shadow_map = calculate_mountain_shadows(height_map, sun_angle)
        current_temp = apply_shadow_effect(current_temp, shadow_map)
        
        # 计算水体效应
        water_bodies = identify_water_bodies(height_map)
        current_humid = update_local_humidity(current_humid, water_bodies)
        
        # 计算地形性降水
        precipitation = calculate_orographic_precipitation(height_map, current_humid)
        current_humid = update_humidity_from_precipitation(current_humid, precipitation)
    
    return current_temp, current_humid


def simulate_ecosystem_dynamics_old(height_map, temp_map, humid_map, iterations=5):
    """模拟生态系统动态，增强地形、温度和湿度之间的相互作用
    
    Args:
        height_map: 高度图
        temp_map: 温度图
        humid_map: 湿度图
        iterations: 迭代次数
        
    Returns:
        更新后的温度图和湿度图
    """
    height, width = height_map.shape
    
    # 创建山体阴影图
    def calculate_mountain_shadows(height_map, sun_angle=(1.0, 1.0, 0.5)):
        """计算山体阴影效应
        sun_angle: (x, y, z)方向的光照向量
        """
        sun_x, sun_y, sun_z = sun_angle
        sun_norm = np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)
        sun_x, sun_y, sun_z = sun_x/sun_norm, sun_y/sun_norm, sun_z/sun_norm
        
        # 计算地形梯度
        gradient_y, gradient_x = np.gradient(height_map)
        
        # 单位化法线向量
        normal_x = -gradient_x
        normal_y = -gradient_y
        normal_z = np.ones_like(height_map)
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm
        
        # 计算光照强度（法线点乘光照方向）
        lighting = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z
        
        # 限制在0到1之间
        shadows = np.clip(lighting, 0, 1)
        return shadows
    
    # 模拟水汽凝结和降水效应
    def calculate_orographic_precipitation(height_map, humid_map):
        """计算地形抬升造成的降水
        根据湿空气上升遇冷形成降水原理
        """
        gradient_y, gradient_x = np.gradient(height_map)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 风吹上坡时湿度凝结形成降水
        # 简化模型，假设风从左到右吹
        upslope = np.zeros_like(height_map)
        upslope[:, 1:] = np.maximum(0, gradient_x[:, 1:])
        
        # 降水强度与坡度和湿度正相关
        precipitation = upslope * humid_map * 0.5
        return precipitation
    
    # 更新温度图和湿度图
    temp_map_new = temp_map.copy()
    humid_map_new = humid_map.copy()
    
    for _ in range(iterations):
        # 计算山体阴影
        shadows = calculate_mountain_shadows(height_map)
        
        # 应用阴影效应到温度图
        # 阴影区域温度降低
        shadow_effect = 0.15 * (shadows - 0.5)
        temp_map_new = temp_map_new + shadow_effect
        
        # 计算降水
        precipitation = calculate_orographic_precipitation(height_map, humid_map_new)
        
        # 降水将减少空气湿度
        humid_map_new = humid_map_new - precipitation * 0.3
        
        # 降水区域周围湿度增加（水分蒸发）
        evaporation = gaussian_filter(precipitation, sigma=2) * 0.2
        humid_map_new = humid_map_new + evaporation
        
        # 找出水体区域（低洼+湿润地区）
        water_bodies = (height_map < np.percentile(height_map, 20)) & (humid_map_new > 0.6)
        
        # 水体对周围湿度的影响
        water_influence = gaussian_filter(water_bodies.astype(float), sigma=3) * 0.1
        humid_map_new = humid_map_new + water_influence
        
        # 水体的热容量效应（降低温差）
        water_temp_effect = gaussian_filter(water_bodies.astype(float), sigma=2) * 0.1
        # 调节温度向平均值靠拢
        mean_temp = np.mean(temp_map_new)
        temp_map_new = temp_map_new * (1 - water_temp_effect) + mean_temp * water_temp_effect
        
        # 确保值在合理范围内
        temp_map_new = np.clip(temp_map_new, 0, 1)
        humid_map_new = np.clip(humid_map_new, 0, 1)
    
    return temp_map_new, humid_map_new
