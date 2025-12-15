import numpy as np
from numba import jit

@jit(nopython=True)
def apply_mountain_range_features(height_map, terrain_types, mountain_type="folded", intensity=1.0):
    """应用特定山脉类型的形态特征
    
    Args:
        height_map: 高度图
        terrain_types: 地形类型图
        mountain_type: 山脉类型 - "folded"(褶皱山脉), "block"(断块山), "volcanic"(火山锥)
        intensity: 特征强度
    """
    height, width = height_map.shape
    result = height_map.copy()
    
    # 计算坡度和朝向用于后续形态处理
    slope_map = np.zeros((height, width))
    aspect_map = np.zeros((height, width))
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 计算x和y方向的梯度
            dx = (height_map[y, x+1] - height_map[y, x-1]) * 0.5
            dy = (height_map[y+1, x] - height_map[y-1, x]) * 0.5
            
            # 计算坡度(角度)
            slope_map[y, x] = np.arctan(np.sqrt(dx*dx + dy*dy)) * (180.0 / np.pi)
            
            # 计算朝向(方位角)
            aspect_map[y, x] = np.arctan2(dy, dx) * (180.0 / np.pi)
            if aspect_map[y, x] < 0:
                aspect_map[y, x] += 360.0
    
    # 对不同山脉类型应用对应的形态特征
    if mountain_type == "folded":  # 褶皱山脉
        # 识别山脊线
        ridges = np.zeros((height, width), dtype=np.bool_)
        for y in range(2, height-2):
            for x in range(2, width-2):
                if terrain_types[y, x] == 0:  # 山区
                    # 检查是否为山脊
                    is_ridge = True
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = x+dx, y+dy
                        if height_map[y, x] < height_map[ny, nx]:
                            is_ridge = False
                            break
                    
                    if is_ridge:
                        ridges[y, x] = True
        
        # 增强山脊特征 - 锯齿状山脊线
        for y in range(2, height-2):
            for x in range(2, width-2):
                if ridges[y, x]:
                    # 增加高度变化，形成锯齿状山脊
                    variation = np.sin(x * 0.5) * np.cos(y * 0.3) * 5.0 * intensity
                    result[y, x] += variation
                    
                    # 应用山脊两侧坡度不对称特性
                    # 迎风坡(20-30°)，背风坡(35-45°)
                    wind_dir = 180  # 假设主风向为南风
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            if dy == 0 and dx == 0:
                                continue
                                
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < height and 0 <= nx < width:
                                # 计算该点相对于山脊的方位
                                angle_to_ridge = np.arctan2(dy, dx) * (180.0 / np.pi)
                                if angle_to_ridge < 0:
                                    angle_to_ridge += 360.0
                                
                                # 计算与风向的夹角
                                angle_diff = abs((angle_to_ridge - wind_dir + 180) % 360 - 180)
                                
                                # 调整坡度
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist > 0:
                                    if angle_diff < 90:  # 迎风坡
                                        # 使坡度更缓(20-30°)
                                        target_slope = (25.0 + np.random.random() * 10.0) * intensity
                                        height_diff = result[y, x] - result[ny, nx]
                                        ideal_diff = np.tan(np.radians(target_slope)) * dist
                                        if height_diff > ideal_diff and height_diff > 0:
                                            result[ny, nx] = result[y, x] - ideal_diff
                                    else:  # 背风坡
                                        # 使坡度更陡(35-45°)
                                        target_slope = (35.0 + np.random.random() * 10.0) * intensity
                                        height_diff = result[y, x] - result[ny, nx]
                                        ideal_diff = np.tan(np.radians(target_slope)) * dist
                                        if height_diff < ideal_diff and height_diff > 0:
                                            result[ny, nx] = result[y, x] - ideal_diff
    
    elif mountain_type == "block":  # 断块山
        # 生成断块山特征：陡峭的断层崖和平坦山顶
        for y in range(2, height-2):
            for x in range(2, width-2):
                if terrain_types[y, x] == 0:  # 山区
                    # 如果是山顶区域，使其更平坦
                    if slope_map[y, x] < 10.0:
                        # 局部平坦化
                        avg_height = 0.0
                        count = 0
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < height and 0 <= nx < width and slope_map[ny, nx] < 10.0:
                                    avg_height += height_map[ny, nx]
                                    count += 1
                        
                        if count > 0:
                            # 稍微平坦化山顶，但保留微起伏
                            result[y, x] = avg_height / count + np.random.random() * 2.0 * intensity
                    
                    # 如果是斜坡区域并且坡度在15-30度之间，可能是断层崖
                    elif 15.0 < slope_map[y, x] < 30.0:
                        # 增加坡度形成陡崖(60-80°)
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                if dy == 0 and dx == 0:
                                    continue
                                    
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    # 计算高度差和距离
                                    height_diff = height_map[y, x] - height_map[ny, nx]
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    
                                    if height_diff > 0 and dist > 0:
                                        # 计算当前坡度
                                        current_slope = np.degrees(np.arctan(height_diff / dist))
                                        
                                        # 如果坡度在特定范围内，增强为断层崖
                                        if 15.0 < current_slope < 40.0:
                                            # 目标坡度在60-80度之间
                                            target_slope = (60.0 + np.random.random() * 20.0) * intensity
                                            ideal_diff = np.tan(np.radians(target_slope)) * dist
                                            
                                            # 调整高度差，使坡度更陡
                                            if height_diff < ideal_diff:
                                                result[ny, nx] = result[y, x] - ideal_diff
    
    elif mountain_type == "volcanic":  # 火山锥
        # 识别可能的火山锥位置
        volcano_centers = []
        for y in range(5, height-5):
            for x in range(5, width-5):
                if terrain_types[y, x] == 0 and slope_map[y, x] < 15.0:
                    # 检查是否周围高度都低于该点
                    is_peak = True
                    for dy in range(-4, 5):
                        for dx in range(-4, 5):
                            if dy == 0 and dx == 0:
                                continue
                            
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if height_map[ny, nx] > height_map[y, x]:
                                    is_peak = False
                                    break
                    
                    if is_peak:
                        # 找到一个可能的火山中心
                        volcano_centers.append((y, x))
        
        # 应用火山锥特征
        for center_y, center_x in volcano_centers:
            # 火山锥半径 - 根据高度动态计算
            radius = int(10 + height_map[center_y, center_x] * 0.3)
            
            # 修改地形形成火山锥体和火山口
            for y in range(center_y-radius, center_y+radius+1):
                for x in range(center_x-radius, center_x+radius+1):
                    if 0 <= y < height and 0 <= x < width:
                        # 计算到中心的距离
                        dist = np.sqrt((y-center_y)**2 + (x-center_x)**2)
                        
                        if dist <= radius:
                            # 火山锥体坡度(5-15°)
                            if dist > radius * 0.2:  # 火山锥体区域
                                # 距离中心的归一化距离
                                norm_dist = dist / radius
                                
                                # 基于距离的高度计算，形成锥体
                                cone_height = height_map[center_y, center_x] * (1.0 - norm_dist**0.5)
                                
                                # 应用锥体形状，但保持一定随机性
                                slope_factor = (5.0 + np.random.random() * 10.0) * intensity
                                ideal_height = height_map[center_y, center_x] - np.tan(np.radians(slope_factor)) * dist
                                
                                # 加权平均确保平滑过渡
                                result[y, x] = (ideal_height * 0.7 + cone_height * 0.3)
                            
                            else:  # 火山口区域
                                # 形成凹陷的火山口
                                crater_depth = height_map[center_y, center_x] * 0.2 * intensity
                                crater_shape = 1.0 - (dist / (radius * 0.2))**2
                                
                                # 火山口边缘更高，中心凹陷
                                if crater_shape > 0.8:  # 火山口边缘
                                    result[y, x] = height_map[center_y, center_x] + 5.0 * intensity
                                else:  # 火山口内部
                                    result[y, x] = height_map[center_y, center_x] - crater_depth * (1.0 - crater_shape)
    
    return result

@jit(nopython=True)
def apply_river_features(height_map, river_map, river_type="v_shaped"):
    """应用特定河流类型的地貌特征
    
    Args:
        height_map: 高度图
        river_map: 河流图(布尔值数组，True表示河流位置)
        river_type: 河流类型 - "v_shaped"(V型谷), "meandering"(曲流河道), "delta"(三角洲)
    """
    height, width = height_map.shape
    result = height_map.copy()
    
    # 对河流及其周围区域应用特定形态特征
    for y in range(2, height-2):
        for x in range(2, width-2):
            if river_map[y, x]:
                # 计算河流走向
                river_dx, river_dy = 0, 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < height and 0 <= nx < width and river_map[ny, nx]:
                            river_dx += dx
                            river_dy += dy
                
                if river_dx != 0 or river_dy != 0:
                    # 河流走向单位向量
                    length = np.sqrt(river_dx**2 + river_dy**2)
                    if length > 0:
                        river_dx /= length
                        river_dy /= length
                    
                    # 河流垂直方向
                    perp_dx, perp_dy = -river_dy, river_dx
                    
                    if river_type == "v_shaped":  # V型谷
                        # 修改河岸坡度，形成V型谷
                        for dist in range(1, 6):
                            for side in [-1, 1]:  # 左右两岸
                                # 沿垂直方向的点
                                nx = int(x + perp_dx * dist * side)
                                ny = int(y + perp_dy * dist * side)
                                
                                if 0 <= ny < height and 0 <= nx < width:
                                    # 形成陡峭的V型谷壁(40-60°坡度)
                                    slope_angle = 40.0 + np.random.random() * 20.0
                                    height_diff = np.tan(np.radians(slope_angle)) * dist
                                    
                                    # 设置新高度，确保高于河床
                                    new_height = result[y, x] + height_diff
                                    if result[ny, nx] < new_height:
                                        result[ny, nx] = new_height
                    
                    elif river_type == "meandering":  # 曲流河道
                        # 计算河道弯曲度 - 通过分析附近河流点的分布
                        curve_strength = 0.0
                        for r in range(2, 5):
                            for angle in range(0, 360, 30):
                                rad = np.radians(angle)
                                cx = int(x + r * np.cos(rad))
                                cy = int(y + r * np.sin(rad))
                                
                                if 0 <= cy < height and 0 <= cx < width and river_map[cy, cx]:
                                    # 计算该点相对于当前点的角度
                                    rel_angle = np.arctan2(cy-y, cx-x)
                                    # 与河流走向的夹角
                                    flow_angle = np.arctan2(river_dy, river_dx)
                                    angle_diff = abs(rel_angle - flow_angle)
                                    
                                    # 角度差大说明河流弯曲
                                    curve_strength += angle_diff
                        
                        # 根据弯曲度应用不同的侵蚀/沉积模式
                        if curve_strength > 1.0:  # 弯曲河道
                            # 凹岸侵蚀(形成陡崖)，凸岸沉积(形成边滩)
                            for dist in range(1, 5):
                                # 凹岸侧 - 侵蚀形成陡崖
                                nx_concave = int(x - perp_dx * dist)
                                ny_concave = int(y - perp_dy * dist)
                                
                                if 0 <= ny_concave < height and 0 <= nx_concave < width:
                                    # 形成近垂直的凹岸(坡度约70°)
                                    concave_height = result[y, x] + np.tan(np.radians(70.0)) * dist
                                    if result[ny_concave, nx_concave] > concave_height:
                                        result[ny_concave, nx_concave] = concave_height
                                
                                # 凸岸侧 - 沉积形成边滩
                                nx_convex = int(x + perp_dx * dist)
                                ny_convex = int(y + perp_dy * dist)
                                
                                if 0 <= ny_convex < height and 0 <= nx_convex < width:
                                    # 形成缓坡边滩(坡度2-5°)
                                    slope_angle = 2.0 + np.random.random() * 3.0
                                    convex_height = result[y, x] + np.tan(np.radians(slope_angle)) * dist
                                    
                                    # 边滩高度平滑过渡
                                    result[ny_convex, nx_convex] = (result[ny_convex, nx_convex] + convex_height) * 0.5
                        
                        else:  # 较直的河道段
                            # 对称的河岸坡度
                            for dist in range(1, 5):
                                for side in [-1, 1]:
                                    nx = int(x + perp_dx * dist * side)
                                    ny = int(y + perp_dy * dist * side)
                                    
                                    if 0 <= ny < height and 0 <= nx < width:
                                        # 形成平缓的河岸(坡度10-15°)
                                        slope_angle = 10.0 + np.random.random() * 5.0
                                        bank_height = result[y, x] + np.tan(np.radians(slope_angle)) * dist
                                        
                                        # 确保连续性
                                        if result[ny, nx] < bank_height:
                                            result[ny, nx] = bank_height
                    
                    elif river_type == "delta":  # 三角洲
                        # 检测是否为三角洲区域 - 通过分析地形高度和河流宽度
                        nearby_water = 0
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    if result[ny, nx] < 15.0:  # 低洼区域(可能是水域)
                                        nearby_water += 1
                        
                        is_delta_region = nearby_water > 10 and result[y, x] < 20.0
                        
                        if is_delta_region:
                            # 修改地形形成三角洲特征 - 极平坦的顶部和缓坡前缘
                            for dy in range(-4, 5):
                                for dx in range(-4, 5):
                                    ny, nx = y+dy, x+dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        dist = np.sqrt(dy**2 + dx**2)
                                        
                                        if dist > 0:
                                            # 距离河流出口的距离
                                            outflow_dir_y, outflow_dir_x = -river_dy, -river_dx  # 河流流向
                                            projection = dy*outflow_dir_y + dx*outflow_dir_x
                                            
                                            if projection > 0:  # 三角洲前缘
                                                # 形成缓坡前缘(坡度1-3°)
                                                slope_angle = 1.0 + np.random.random() * 2.0
                                                delta_height = result[y, x] - np.tan(np.radians(slope_angle)) * projection
                                                
                                                # 确保平滑过渡
                                                result[ny, nx] = (result[ny, nx] + delta_height) * 0.5
                                            
                                            else:  # 三角洲顶部
                                                # 形成极平坦表面(坡度<0.1°)
                                                flat_height = result[y, x] - np.tan(np.radians(0.1)) * abs(projection)
                                                
                                                # 应用平坦化
                                                result[ny, nx] = (result[ny, nx] * 0.2 + flat_height * 0.8)
    
    return result

@jit(nopython=True)
def apply_glacial_features(height_map, terrain_types, feature_type="u_shaped"):
    """应用冰川地貌特征
    
    Args:
        height_map: 高度图
        terrain_types: 地形类型图
        feature_type: 特征类型 - "u_shaped"(U型谷), "cirque"(冰斗), "drumlin"(鼓丘)
    """
    height, width = height_map.shape
    result = height_map.copy()
    
    # 计算坡度和坡向
    slope_map = np.zeros((height, width))
    aspect_map = np.zeros((height, width))
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            dx = (height_map[y, x+1] - height_map[y, x-1]) * 0.5
            dy = (height_map[y+1, x] - height_map[y-1, x]) * 0.5
            slope_map[y, x] = np.sqrt(dx*dx + dy*dy)
            aspect_map[y, x] = np.arctan2(dy, dx)
    
    if feature_type == "u_shaped":  # U型谷
        # 识别可能的谷地位置
        valley_points = []
        for y in range(2, height-2):
            for x in range(2, width-2):
                # 寻找山区中较低的区域
                if terrain_types[y, x] == 0 and slope_map[y, x] < 0.2:
                    # 检查是否两侧高度较高(可能是谷地)
                    left_higher = False
                    right_higher = False
                    
                    # 检查东西方向
                    for dx in range(-5, -1):
                        if 0 <= x+dx < width and height_map[y, x+dx] > height_map[y, x] + 10:
                            left_higher = True
                            break
                    
                    for dx in range(2, 6):
                        if 0 <= x+dx < width and height_map[y, x+dx] > height_map[y, x] + 10:
                            right_higher = True
                            break
                    
                    # 或检查南北方向
                    north_higher = False
                    south_higher = False
                    
                    for dy in range(-5, -1):
                        if 0 <= y+dy < height and height_map[y+dy, x] > height_map[y, x] + 10:
                            north_higher = True
                            break
                    
                    for dy in range(2, 6):
                        if 0 <= y+dy < height and height_map[y+dy, x] > height_map[y, x] + 10:
                            south_higher = True
                            break
                    
                    # 如果两侧都较高，可能是谷地
                    if (left_higher and right_higher) or (north_higher and south_higher):
                        valley_points.append((y, x))
        
        # 应用U型谷特征
        for center_y, center_x in valley_points:
            # 确定谷地主方向
            if abs(aspect_map[center_y, center_x]) < np.pi/4 or abs(aspect_map[center_y, center_x]) > np.pi*3/4:
                # 东西走向谷地
                valley_dx, valley_dy = 1, 0
            else:
                # 南北走向谷地
                valley_dx, valley_dy = 0, 1
            
            # 谷地垂直方向
            perp_dx, perp_dy = -valley_dy, valley_dx
            
            # U型谷宽度
            valley_width = int(10 + np.random.random() * 5)
            
            # 修改地形形成U型谷
            for dist in range(-valley_width, valley_width+1):
                # 沿垂直方向的点
                nx = center_x + perp_dx * dist
                ny = center_y + perp_dy * dist
                
                if 0 <= ny < height and 0 <= nx < width:
                    # U形谷底较平坦
                    if abs(dist) < valley_width * 0.3:
                        # 平坦谷底，稍有起伏
                        result[ny, nx] = result[center_y, center_x] + np.random.random() * 2.0
                    else:
                        # 抛物线形谷壁，坡度随距离增加
                        # U = a*(x^2) + b 形式的抛物线，a控制陡峭度
                        steepness = 0.05 + np.random.random() * 0.02
                        normalized_dist = (abs(dist) - valley_width * 0.3) / (valley_width * 0.7)
                        wall_height = steepness * normalized_dist**2 * 100.0
                        
                        # 设置最终高度
                        target_height = result[center_y, center_x] + wall_height
                        if result[ny, nx] < target_height:
                            result[ny, nx] = target_height
                        
                        # 在谷肩处(谷壁上部)添加陡坎
                        if abs(dist) > valley_width * 0.8:
                            shoulder_height = wall_height * 1.2
                            target_height = result[center_y, center_x] + shoulder_height
                            if result[ny, nx] < target_height:
                                result[ny, nx] = target_height
    
    elif feature_type == "cirque":  # 冰斗
        # 识别潜在的冰斗位置 - 通常在山区高海拔背风坡
        cirque_centers = []
        for y in range(5, height-5):
            for x in range(5, width-5):
                if terrain_types[y, x] == 0 and height_map[y, x] > 60.0:
                    # 检查是否一面有较高的山壁，另一面开口较低
                    back_higher = False
                    front_lower = False
                    
                    # 使用坡向来确定方向
                    # 坡向指向低处方向
                    angle = aspect_map[y, x]
                    
                    # 检查"背面"方向是否有高地
                    back_dx = -np.cos(angle) * 4
                    back_dy = -np.sin(angle) * 4
                    back_x = int(x + back_dx)
                    back_y = int(y + back_dy)
                    
                    if 0 <= back_y < height and 0 <= back_x < width:
                        if height_map[back_y, back_x] > height_map[y, x] + 15.0:
                            back_higher = True
                    
                    # 检查"前面"方向是否地势下降
                    front_dx = np.cos(angle) * 4
                    front_dy = np.sin(angle) * 4
                    front_x = int(x + front_dx)
                    front_y = int(y + front_dy)
                    
                    if 0 <= front_y < height and 0 <= front_x < width:
                        if height_map[front_y, front_x] < height_map[y, x] - 5.0:
                            front_lower = True
                    
                    # 如果满足条件，认为是潜在的冰斗位置
                    if back_higher and front_lower:
                        cirque_centers.append((y, x, angle))
        
        # 应用冰斗特征
        for center_y, center_x, angle in cirque_centers:
            # 冰斗半径
            radius = int(5 + np.random.random() * 5)
            
            # 冰斗方向向量(指向开口方向)
            dir_dx = np.cos(angle)
            dir_dy = np.sin(angle)
            
            # 应用冰斗形态
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    ny, nx = center_y + dy, center_x + dx
                    
                    if 0 <= ny < height and 0 <= nx < width:
                        # 计算到中心的距离
                        dist = np.sqrt(dy**2 + dx**2)
                        
                        if dist <= radius:
                            # 计算点相对于中心的方向
                            if dist > 0:
                                point_dx, point_dy = dx/dist, dy/dist
                                
                                # 与开口方向的夹角
                                dot_product = point_dx*dir_dx + point_dy*dir_dy
                                # 夹角余弦值，1表示同向，-1表示反向
                                
                                if dot_product < -0.3:  # 背壁方向
                                    # 形成陡峭的后壁(坡度>50°)
                                    norm_dist = dist / radius
                                    wall_height = 30.0 * (1.0 - norm_dist**0.5)
                                    
                                    new_height = height_map[center_y, center_x] + wall_height
                                    if result[ny, nx] < new_height:
                                        result[ny, nx] = new_height
                                
                                elif dot_product > 0.3:  # 开口方向
                                    # 形成缓坡出口
                                    norm_dist = dist / radius
                                    exit_depth = 10.0 * norm_dist
                                    
                                    new_height = height_map[center_y, center_x] - exit_depth
                                    if result[ny, nx] > new_height:
                                        result[ny, nx] = new_height
                                
                                else:  # 侧壁
                                    # 形成弧形侧壁
                                    norm_dist = dist / radius
                                    side_height = 15.0 * (1.0 - norm_dist**2)
                                    
                                    new_height = height_map[center_y, center_x] + side_height
                                    result[ny, nx] = new_height
                            
                            else:  # 中心点
                                # 形成平坦底部
                                result[ny, nx] = height_map[center_y, center_x] - 5.0
    
    return result

# 主函数：
def apply_realistic_terrain_features(height_map, terrain_types, river_map=None):
    """应用真实地形特征到高度图
    
    Args:
        height_map: 高度图
        terrain_types: 地形类型图(0=山区, 1=丘陵, 2=平原, 3=高原)
        river_map: 河流图(布尔值数组，True表示河流位置)
    
    Returns:
        应用了真实地形特征的高度图
    """
    # 创建结果数组
    result = height_map.copy()
    
    # 1. 应用山脉特征
    # 随机选择山脉类型(实际应用中可以让用户选择或混合使用)
    mountain_types = ["folded", "block", "volcanic"]
    selected_type = mountain_types[np.random.randint(0, len(mountain_types))]
    
    result = apply_mountain_range_features(result, terrain_types, mountain_type=selected_type)
    
    # 2. 应用河流特征(如果提供了河流图)
    if river_map is not None:
        river_types = ["v_shaped", "meandering", "delta"]
        # 根据地区海拔选择不同的河流类型
        for y in range(len(river_map)):
            for x in range(len(river_map[0])):
                if river_map[y, x]:
                    # 根据高度选择河流类型
                    if height_map[y, x] > 60.0:  # 高海拔
                        river_type = "v_shaped"
                    elif height_map[y, x] > 20.0:  # 中海拔
                        river_type = "meandering"
                    else:  # 低海拔
                        river_type = "delta"
                    
                    # 在河流周围应用特征
                    local_slice = np.zeros_like(river_map)
                    y_start = max(0, y-5)
                    y_end = min(len(river_map), y+6)
                    x_start = max(0, x-5)
                    x_end = min(len(river_map[0]), x+6)
                    
                    local_slice[y_start:y_end, x_start:x_end] = river_map[y_start:y_end, x_start:x_end]
                    result = apply_river_features(result, local_slice, river_type=river_type)
    
    # 3. 在高海拔区域应用冰川地貌特征
    # 随机选择启用或禁用(根据气候条件)
    if np.random.random() > 0.5:  # 50%的概率应用冰川特征
        # 随机选择特征类型
        glacial_types = ["u_shaped", "cirque"]
        selected_type = glacial_types[np.random.randint(0, len(glacial_types))]
        
        result = apply_glacial_features(result, terrain_types, feature_type=selected_type)
    
    return result