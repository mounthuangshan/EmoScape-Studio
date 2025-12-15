import numpy as np
from scipy.ndimage import gaussian_filter

def create_biome_transitions(biome_map, height_map, temp_map, humid_map, biome_data):
    """创建生物群系过渡区，使不同生物群系之间的边界更加自然
    
    Args:
        biome_map: 生物群系映射数组
        height_map: 高度图
        temp_map: 温度图
        humid_map: 湿度图
        biome_data: 生物群系数据
        
    Returns:
        带有过渡区的新生物群系映射
    """
    height, width = biome_map.shape
    
    # 创建扩展生物群系属性图
    biome_props = {}
    for biome_id in np.unique(biome_map):
        if biome_id in biome_data:
            biome_props[biome_id] = {
                'min_height': biome_data[biome_id].get('min_height', 0),
                'max_height': biome_data[biome_id].get('max_height', 1),
                'min_temp': biome_data[biome_id].get('min_temp', 0),
                'max_temp': biome_data[biome_id].get('max_temp', 1),
                'min_humid': biome_data[biome_id].get('min_humid', 0),
                'max_humid': biome_data[biome_id].get('max_humid', 1)
            }
    
    # 计算每个生物群系的模糊边界
    fuzzy_biomes = {}
    for biome_id in biome_props:
        # 创建该生物群系的二进制掩码
        mask = (biome_map == biome_id).astype(float)
        # 应用高斯模糊以创建过渡区
        fuzzy = gaussian_filter(mask, sigma=1.5)
        fuzzy_biomes[biome_id] = fuzzy
    
    # 创建过渡区生物群系地图
    transition_map = np.zeros_like(biome_map)
    
    # 边界距离缓冲区
    edge_distance = np.zeros_like(biome_map, dtype=float)
    
    # 计算到边界的距离
    for biome_id in fuzzy_biomes:
        # 找出该生物群系的内部区域(不包括边界)
        inner = gaussian_filter(fuzzy_biomes[biome_id], sigma=0.5) > 0.9
        # 找出边界区域
        edge = (fuzzy_biomes[biome_id] > 0.1) & (~inner)
        # 更新距离图
        edge_distance[edge] = 1.0
    
    # 距离变换，计算每个点到最近边界的距离
    from scipy.ndimage import distance_transform_edt
    distance_to_edge = distance_transform_edt(1 - edge_distance)
    
    # 应用过渡规则
    transition_width = 3  # 过渡区宽度
    
    # 处理每个像素
    for y in range(height):
        for x in range(width):
            # 如果是边界附近的区域
            if distance_to_edge[y, x] < transition_width:
                # 找出影响该点的生物群系
                influences = {}
                for biome_id in fuzzy_biomes:
                    influence = fuzzy_biomes[biome_id][y, x]
                    if influence > 0.05:  # 影响力阈值
                        influences[biome_id] = influence
                
                # 如果有多个影响，则创建混合生物群系
                if len(influences) > 1:
                    # 归一化影响值
                    total = sum(influences.values())
                    for biome_id in influences:
                        influences[biome_id] /= total
                    
                    # 使用原始生物群系的地形条件进行加权决策
                    height_value = height_map[y, x]
                    temp_value = temp_map[y, x]
                    humid_value = humid_map[y, x]
                    
                    # 计算最匹配的生物群系
                    best_match = None
                    best_score = -1
                    
                    for biome_id, influence in influences.items():
                        if biome_id in biome_props:
                            props = biome_props[biome_id]
                            
                            # 计算环境匹配得分
                            height_match = 1.0 - min(1.0, abs(height_value - (props['min_height'] + props['max_height'])/2) / 
                                                  ((props['max_height'] - props['min_height'])/2 + 1e-10))
                            
                            temp_match = 1.0 - min(1.0, abs(temp_value - (props['min_temp'] + props['max_temp'])/2) / 
                                                ((props['max_temp'] - props['min_temp'])/2 + 1e-10))
                            
                            humid_match = 1.0 - min(1.0, abs(humid_value - (props['min_humid'] + props['max_humid'])/2) / 
                                                 ((props['max_humid'] - props['min_humid'])/2 + 1e-10))
                            
                            # 综合得分 (加权环境匹配 + 生物群系影响力)
                            score = (height_match * 0.3 + temp_match * 0.35 + humid_match * 0.35) * 0.7 + influence * 0.3
                            
                            if score > best_score:
                                best_score = score
                                best_match = biome_id
                    
                    if best_match is not None:
                        transition_map[y, x] = best_match
                else:
                    # 单一生物群系区域
                    transition_map[y, x] = biome_map[y, x]
            else:
                # 非过渡区，保持原始生物群系
                transition_map[y, x] = biome_map[y, x]
    
    return transition_map