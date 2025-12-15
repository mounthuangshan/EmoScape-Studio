#from __future__ import annotations
#标准库
import os
import random
import tempfile
import heapq
import itertools

#数据处理与科学计算
import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy import ndimage
from scipy.ndimage import convolve, gaussian_filter, minimum_filter, maximum_filter
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发

#其他工具
import neat

#项目文件
from utils.tools import *


################
#使用NEAT优化布局
################
def evaluate_single_genome(genome, config, building_data):
    """优化后的评估函数：通过空间采样和启发式搜索加速"""
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    grid_size = 100
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    layout_score = 0
    placed_buildings = []
    building_count = len(building_data)

    # 预排序建筑（按面积从大到小）
    sorted_indices = sorted(
        range(building_count),
        key=lambda i: building_data[i]['width'] * building_data[i]['height'],
        reverse=True
    )

    # 参数配置
    MAX_SAMPLES = 50       # 每个建筑的位置采样次数
    STEP_SIZE = 10         # 网格搜索步长（原为5）

    for building_idx in sorted_indices:
        b = building_data[building_idx]
        best_score = -float('inf')
        best_pos = None

        # 生成候选位置（带随机性的网格采样）
        max_x = grid_size - b['width']
        max_y = grid_size - b['height']
        x_samples = range(0, max_x + 1, STEP_SIZE)
        y_samples = range(0, max_y + 1, STEP_SIZE)
        candidates = list(itertools.product(x_samples, y_samples))
        random.shuffle(candidates)  # 引入随机性避免局部最优

        # 评估前N个候选位置
        for x, y in candidates[:MAX_SAMPLES]:
            if not _is_valid_position(grid, x, y, b):
                continue

            # 构建输入向量（标准化到0-1范围）
            input_vec = [
                x / grid_size,
                y / grid_size,
                * _collect_layout_state(placed_buildings, grid_size),
                * b['features']
            ]
            
            # 神经网络评估
            score = net.activate(input_vec)[0]
            
            if score > best_score:
                best_score = score
                best_pos = (x, y)

        # 放置建筑或给予惩罚
        if best_pos:
            x, y = best_pos
            _place_building(grid, x, y, b)
            placed_buildings.append({
                "x": x, "y": y, 
                "width": b['width'], 
                "height": b['height']
            })
            layout_score += best_score * (1 + len(placed_buildings)/building_count)
        else:
            layout_score -= 1000  # 惩罚项
            break

    return layout_score

def run_neat_for_content_layout(buildings, config_path=None, population_size=50, generations=3):
    """
    使用NEAT算法优化建筑布局
    
    Args:
        buildings: 需要布局的建筑物列表
        config_path: NEAT配置文件路径，默认使用内置配置
        population_size: 种群大小
        generations: 进化代数
        
    Returns:
        tuple: (最优布局顺序, 包含附加对象的布局信息)
    """
    
    if len(buildings) < 2:
        return list(range(len(buildings))), {"objects":[]}
    
    # 提取建筑物特征
    building_features = _extract_building_features(buildings)
    
        # 计算输入向量维度 ---------------------------------------------------
    # 输入结构: position_input(2) + layout_state(5) + current_building_features(n)
    if building_features:
        features_per_building = len(building_features[0]['features'])
    else:
        features_per_building = 0
        
    input_length = 2 + 5 + features_per_building  # 确定实际输入维度
    
    # 创建临时NEAT配置文件（传递正确输入长度）
    if config_path is None or not os.path.exists(config_path):
        config_path = _create_temp_neat_config(input_length)  # 传入实际输入长度
    
    # 定义符合NEAT接口的评估函数
    def eval_genomes(genomes, config):
        """标准化的评估函数签名"""
        for genome_id, genome in genomes:
            # 在闭包内直接访问building_features
            genome.fitness = evaluate_single_genome(genome, config, building_features)
    
    # 设置NEAT配置
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_path
    )

    # 配置种群
    config.pop_size = population_size

    # 创建进化过程
    pop = neat.Population(config)

    # 添加输出报告器
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # 创建适应度函数
    eval_function = eval_genomes

    # 运行进化
    best_genome = pop.run(eval_function, generations)

    # 使用最佳基因组生成最终布局
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    final_order, final_positions = _generate_final_layout(net, building_features)

    # 构建额外对象（例如路径、装饰物等）
    additional_objects = _generate_additional_objects(final_positions)

    return final_order, {"objects": additional_objects}

def _extract_building_features(buildings):
    """提取建筑物的关键特征"""
    features = []
    
    for building in buildings:
        # 统一数据结构：如果是元组则转换为字典
        if isinstance(building, tuple):
            building = convert_tuple_to_dict(building)
        # 强制转换为整数并校验有效性
        try:
            # 先转换为浮点数再取整，处理字符串输入
            width = int(round(float(building.get('width', 10))))
            width = max(1, width)  # 确保最小为1
        except (ValueError, TypeError):
            width = 10
            
        try:
            height = int(round(float(building.get('height', 10))))
            height = max(1, height)
        except (ValueError, TypeError):
            height = 10
        
        # 提取建筑类型并编码
        building_type = building.get('type', 'default')
        type_encoding = _encode_building_type(building_type)
        
        # 提取并转换其他可能的特征
        importance = float(building.get('importance', 5)) / 10.0
        
        # 合并所有特征
        building_features = {
            'width': width,
            'height': height,
            'features': type_encoding + [importance, width/30.0, height/30.0]
        }
        features.append(building_features)
    
    return features

def _encode_building_type(building_type):
    """将建筑类型编码为特征向量"""
    # 建筑类型编码表（可扩展）
    type_encodings = {
        'residential': [1, 0, 0, 0, 0],
        'commercial': [0, 1, 0, 0, 0],
        'industrial': [0, 0, 1, 0, 0],
        'public': [0, 0, 0, 1, 0],
        'special': [0, 0, 0, 0, 1],
        'default': [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    
    return type_encodings.get(building_type, type_encodings['default'])

def _is_valid_position(grid, x, y, building):
    """检查建筑物放置位置是否有效（无重叠）"""
    width, height = building['width'], building['height']
    
    # 检查是否超出网格边界
    if x < 0 or y < 0 or x + width >= grid.shape[0] or y + height >= grid.shape[1]:
        return False
        
    # 检查是否与现有建筑重叠
    for i in range(width):
        for j in range(height):
            if grid[x + i, y + j]:
                return False
                
    return True

def _place_building(grid, x, y, building):
    """在网格上放置建筑物"""
    width, height = building['width'], building['height']
    
    for i in range(width):
        for j in range(height):
            grid[x + i, y + j] = True

def _collect_layout_state(placed_buildings, grid_size):
    """收集当前布局状态作为神经网络输入"""
    if not placed_buildings:
        return [0.0] * 5
    
    # 计算布局重心
    total_x = sum(b['x'] + b['width']/2 for b in placed_buildings)
    total_y = sum(b['y'] + b['height']/2 for b in placed_buildings)
    center_x = total_x / len(placed_buildings) / grid_size
    center_y = total_y / len(placed_buildings) / grid_size
    
    covered_area = sum(b['width'] * b['height'] for b in placed_buildings)
    density = covered_area / (grid_size * grid_size)
    
    min_x = min(b['x'] for b in placed_buildings) / grid_size
    max_x = max(b['x'] + b['width'] for b in placed_buildings) / grid_size
    
    return [center_x, center_y, density, min_x, max_x]

######################################################################
#在当前的代码实现中，布局质量的评估完全依赖于神经网络的输出 score，
# 而不是通过这个独立的函数计算。神经网络通过学习输入特征（位置、布局状态、建筑特征等）
# 来预测每个位置的得分，最终的布局分数 layout_score 是这些得分的加权累加。
# 这种方法取代了传统的、基于规则的评估函数（如 _calculate_layout_quality）
####################################################################
def _calculate_layout_quality(placed_buildings):
    """计算当前布局的质量分数"""
    if len(placed_buildings) <= 1:
        return 0.0
    
    score = 0.0
    
    # 评估建筑物间距
    for i in range(len(placed_buildings)):
        for j in range(i+1, len(placed_buildings)):
            b1 = placed_buildings[i]
            b2 = placed_buildings[j]
            
            # 计算两建筑中心点距离
            center1_x = b1['x'] + b1['width']/2
            center1_y = b1['y'] + b1['height']/2
            center2_x = b2['x'] + b2['width']/2
            center2_y = b2['y'] + b2['height']/2
            
            distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
            
            # 评估距离合理性（既不太近也不太远）
            ideal_distance = (b1['width'] + b1['height'] + b2['width'] + b2['height'])/4
            distance_score = 1.0 / (1.0 + abs(distance - ideal_distance) / ideal_distance)
            
            # 考虑建筑物类型关系
            relationship_score = _evaluate_building_relationship(b1['features'], b2['features'])
            
            score += distance_score * relationship_score
    
    # 评估整体布局紧凑性
    compactness = _evaluate_compactness(placed_buildings)
    score += compactness * 2.0
    
    # 评估美学分布（例如对称性）
    aesthetics = _evaluate_aesthetics(placed_buildings)
    score += aesthetics * 1.5
    
    return score

def _evaluate_building_relationship(features1, features2):
    """评估两种建筑类型之间的关系适宜性"""
    # 简化：根据建筑类型编码计算相似度或互补性
    sim = sum(abs(a - b) for a, b in zip(features1[:5], features2[:5]))
    return 1.0 / (1.0 + sim)

def _evaluate_compactness(placed_buildings):
    """评估布局的紧凑性"""
    if not placed_buildings:
        return 0.0
        
    # 计算边界盒大小
    min_x = min(b['x'] for b in placed_buildings)
    min_y = min(b['y'] for b in placed_buildings)
    max_x = max(b['x'] + b['width'] for b in placed_buildings)
    max_y = max(b['y'] + b['height'] for b in placed_buildings)
    
    boundary_area = (max_x - min_x) * (max_y - min_y)
    building_area = sum(b['width'] * b['height'] for b in placed_buildings)
    
    if boundary_area == 0:
        return 1.0
        
    # 紧凑性：建筑面积与边界盒面积的比值
    return building_area / boundary_area

def _evaluate_aesthetics(placed_buildings):
    """评估布局的美学价值"""
    if len(placed_buildings) < 3:
        return 0.5
    
    # 计算布局重心
    total_x = sum(b['x'] + b['width']/2 for b in placed_buildings)
    total_y = sum(b['y'] + b['height']/2 for b in placed_buildings)
    center_x = total_x / len(placed_buildings)
    center_y = total_y / len(placed_buildings)
    
    # 计算与重心的距离方差（越低越均衡）
    dist_variance = sum(((b['x'] + b['width']/2 - center_x)**2 + 
                         (b['y'] + b['height']/2 - center_y)**2) 
                        for b in placed_buildings) / len(placed_buildings)
    
    # 方差越小，分数越高
    return 1.0 / (1.0 + 0.01 * dist_variance)


def _generate_final_layout(net, building_features):
    """优化后的最终布局生成：通过空间启发式搜索加速"""
    grid_size = 100
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    placement_order = []
    positions = [] 

    # 预排序前进行尺寸调整
    adjusted_features = []
    for b in building_features:
        # 创建副本避免修改原始数据
        adjusted = b.copy()
        original_w = adjusted['width']
        original_h = adjusted['height']
        
        # 动态调整超限建筑的尺寸（保持宽高比）
        if original_w > grid_size or original_h > grid_size:
            scale = min(grid_size/original_w, grid_size/original_h)
            adjusted['width'] = max(1, int(original_w * scale))
            adjusted['height'] = max(1, int(original_h * scale))
            print(f"Adjusted building from {original_w}x{original_h} to {adjusted['width']}x{adjusted['height']}")

        adjusted_features.append(adjusted)

    # 按调整后的尺寸排序（面积从大到小）
    sorted_indices = sorted(
        range(len(adjusted_features)),
        key=lambda i: adjusted_features[i]['width'] * adjusted_features[i]['height'],
        reverse=True
    )
    
    # 搜索参数配置
    STEP_SIZES = [10, 5]
    MAX_SAMPLES = [50, 100]

    for building_idx in sorted_indices:
        b = adjusted_features[building_idx]
        max_x = max(0, grid_size - b['width'])  # 确保非负
        max_y = max(0, grid_size - b['height'])
        
        best_pos = None
        best_score = -float('inf')
        found = False
        
        # 获取当前布局状态（需要完整字典数据）
        current_inputs = _collect_layout_state(positions, grid_size)  # 传入字典列表
        
        # 分阶段搜索
        candidate_heap = []
        for stage in range(len(STEP_SIZES)):
            step = STEP_SIZES[stage]
            max_samples = MAX_SAMPLES[stage]
            
            # 生成候选位置（从中心向外辐射）
            max_x = grid_size - b['width']
            max_y = grid_size - b['height']
            center_x, center_y = grid_size//2, grid_size//2
            
            # 生成坐标序列（中心优先）
            x_coords = sorted(
                range(0, max_x, step), 
                key=lambda x: abs(x - center_x)
            )
            y_coords = sorted(
                range(0, max_y, step), 
                key=lambda y: abs(y - center_y)
            )
            
            # 生成候选并加入优先队列
            for x, y in itertools.product(x_coords, y_coords):
                if _is_valid_position(grid, x, y, b):
                    heuristic = -(abs(x - center_x) + abs(y - center_y))
                    heapq.heappush(candidate_heap, (heuristic, x, y))
            
            # 评估前N个候选
            evaluated = 0
            while candidate_heap and evaluated < max_samples:
                _, x, y = heapq.heappop(candidate_heap)
                if not _is_valid_position(grid, x, y, b):
                    continue
                
                # 构建输入向量（使用字典中的特征）
                input_vec = [
                    x / grid_size, 
                    y / grid_size,
                    *current_inputs,
                    *b['features']
                ]
                
                # 神经网络评估
                score = net.activate(input_vec)[0]
                
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
                    found = True
                
                evaluated += 1
                if evaluated >= max_samples:
                    break
                
            if found:
                break
        
        # 最终回退策略（随机采样）
        if not found:
            fallback_samples = 200
            max_x = grid_size - b['width']
            max_y = grid_size - b['height']
            for _ in range(fallback_samples):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                if _is_valid_position(grid, x, y, b):
                    best_pos = (x, y)
                    found = True
                    break
        
        if not found:
            continue
            
        # 放置建筑
        if found:
            x, y = best_pos
            _place_building(grid, x, y, b)
            placement_order.append(building_idx)
            # 存储完整建筑信息字典
            positions.append({
                'x': x,
                'y': y,
                'width': b['width'],
                'height': b['height'],
                'features': b['features']  # 保留特征用于后续评估
            })
    
    return placement_order, positions  # positions现在是字典列表

def _generate_additional_objects(positions):
    """基于建筑物布局生成额外装饰物和连接元素"""
    objects = []
    
    # 生成建筑物之间的路径
    paths = _generate_paths(positions)
    objects.extend(paths)
    
    # 添加装饰性元素
    decorative = _generate_decorative_elements(positions)
    objects.extend(decorative)
    
    # 添加功能性元素
    functional = _generate_functional_elements(positions)
    objects.extend(functional)
    
    return objects

def _generate_paths(positions):
    """生成建筑物之间的路径网络"""
    if len(positions) < 2:
        return []
    
    # 构建距离矩阵
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # 从字典中提取坐标和尺寸
            pos_i = positions[i]
            x1, y1 = pos_i['x'], pos_i['y']
            w1, h1 = pos_i['width'], pos_i['height']
            
            pos_j = positions[j]
            x2, y2 = pos_j['x'], pos_j['y']
            w2, h2 = pos_j['width'], pos_j['height']
            
            # 计算中心点距离
            c1x, c1y = x1 + w1/2, y1 + h1/2
            c2x, c2y = x2 + w2/2, y2 + h2/2
            distance = ((c1x - c2x)**2 + (c1y - c2y)**2)**0.5
            
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    # 创建最小生成树并生成路径
    mst = minimum_spanning_tree(dist_matrix).toarray()
    paths = []
    
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                pos_i = positions[i]
                pos_j = positions[j]
                
                # 计算连接路径端点
                c1x = pos_i['x'] + pos_i['width']/2
                c1y = pos_i['y'] + pos_i['height']/2
                c2x = pos_j['x'] + pos_j['width']/2
                c2y = pos_j['y'] + pos_j['height']/2
                
                paths.append({
                    "type": "path",
                    "x1": int(c1x),
                    "y1": int(c1y),
                    "x2": int(c2x),
                    "y2": int(c2y),
                    "width": 2
                })
    
    return paths

def _generate_decorative_elements(positions):
    """在布局中添加装饰元素（已适配字典数据结构）"""
    
    decorations = []
    
    # 遍历所有建筑位置（每个位置是字典）
    for i, pos in enumerate(positions):
        # 从字典中提取建筑信息
        x = pos['x']
        y = pos['y']
        width = pos['width']
        height = pos['height']
        importance = pos.get('importance', 5)  # 重要度现在应包含在位置数据中
        
        if importance > 7:  # 高重要度建筑
            # 添加雕像（放置在建筑右侧外）
            statue_x = x + width + random.randint(1, 3)
            statue_y = y + random.randint(0, height)
            decorations.append({
                "type": "special_statue",
                "x": statue_x,
                "y": statue_y
            })
            
            # 添加装饰性植物（环绕建筑周围）
            for _ in range(2):
                # 生成位置时考虑建筑尺寸
                plant_x = x + random.randint(-3, width + 3)
                plant_y = y + random.randint(-3, height + 3)
                
                # 检查是否在建筑区域内
                if not (x <= plant_x <= x + width and y <= plant_y <= y + height):
                    decorations.append({
                        "type": "decorative_plant",
                        "x": plant_x,
                        "y": plant_y,
                        "size": random.choice(["small", "medium"])
                    })
    
    return decorations

def _generate_functional_elements(positions):
    """添加功能性元素（已适配字典数据结构）"""
    elements = []
    
    if len(positions) < 1:
        return elements
    
    # 计算整体布局的中心点（直接从字典提取数据）
    total_x = sum(pos['x'] + pos['width']/2 for pos in positions)
    total_y = sum(pos['y'] + pos['height']/2 for pos in positions)
    center_x = total_x / len(positions)
    center_y = total_y / len(positions)
    
    # 在中心附近添加水井
    elements.append({
        "type": "well",
        "x": int(center_x),
        "y": int(center_y)
    })
    
    # 计算布局边界
    min_x = min(pos['x'] for pos in positions)
    min_y = min(pos['y'] for pos in positions)
    max_x = max(pos['x'] + pos['width'] for pos in positions)
    max_y = max(pos['y'] + pos['height'] for pos in positions)
    
    # 生成入口候选位置
    entrance_positions = [
        (min_x - 5, (min_y + max_y)//2),  # 左边界中点
        (max_x + 5, (min_y + max_y)//2),   # 右边界中点
        ((min_x + max_x)//2, min_y - 5),   # 下边界中点
        ((min_x + max_x)//2, max_y + 5)    # 上边界中点
    ]
    
    # 随机选择最佳入口
    best_entrance = random.choice(entrance_positions)
    elements.append({
        "type": "entrance",
        "x": best_entrance[0],
        "y": best_entrance[1],
        "width": 3,
        "height": 3
    })
    
    return elements

def _create_temp_neat_config(input_length):
    """创建临时NEAT配置文件"""
    
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100.0
pop_size              = 50
reset_on_extinction   = True

[DefaultGenome]
num_inputs              = {input_length}
num_hidden              = 16
num_outputs             = 1
initial_connection      = partial_direct 0.5
feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 1.0
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
node_add_prob           = 0.2
node_delete_prob        = 0.2
activation_default      = sigmoid
activation_options      = sigmoid
activation_mutate_rate  = 0.1
aggregation_default     = sum
aggregation_options     = sum
aggregation_mutate_rate = 0.0
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_replace_rate       = 0.1
bias_mutate_rate        = 0.7
bias_mutate_power       = 0.5
bias_max_value          = 30.0
bias_min_value          = -30.0
response_init_mean      = 1.0
response_init_stdev     = 0.1
response_replace_rate   = 0.1
response_mutate_rate    = 0.1
response_mutate_power   = 0.1
response_max_value      = 30.0
response_min_value      = -30.0

weight_max_value        = 30
weight_min_value        = -30
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
weight_mutate_power     = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cfg', mode='w+')
    temp_file.write(config_text)
    temp_file.close()
    
    return temp_file.name