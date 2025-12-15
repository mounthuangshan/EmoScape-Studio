import numpy as np
from opensimplex import OpenSimplex
import random
from enum import IntEnum

class UndergroundContentType(IntEnum):
    """地下内容类型枚举"""
    EMPTY = 0
    SOIL = 1
    ROCK = 2
    CAVE = 3
    TUNNEL = 4
    WATER = 5
    LAVA = 6
    UNDERGROUND_RIVER = 7
    CRYSTAL_CAVE = 8
    FUNGAL_CAVE = 9
    ANCIENT_RUINS = 10
    ABANDONED_MINE = 11

class MineralType(IntEnum):
    """矿物类型枚举"""
    NONE = 0
    COAL = 1
    IRON = 2
    COPPER = 3
    GOLD = 4
    SILVER = 5
    DIAMOND = 6
    EMERALD = 7
    RUBY = 8
    SAPPHIRE = 9
    CRYSTAL = 10
    OBSIDIAN = 11
    MYTHRIL = 12  # 幻想矿物
    ADAMANTINE = 13  # 幻想矿物

def generate_underground_layers(map_data, config, logger, progress_callback=None):
    """生成地下层系统
    
    Args:
        map_data: MapData实例
        config: 生成配置
        logger: 日志记录器
        progress_callback: 进度回调函数
    """
    width = map_data.width
    height = map_data.height
    
    # 从配置获取地下层深度，默认3层
    depth = config.get("underground_depth", 3)
    
    # 创建地下层
    logger.log(f"创建{depth}层地下结构...")
    underground_layers = map_data.create_underground_layers(depth)
    
    if progress_callback:
        progress_callback(0.1, f"创建{depth}层地下结构...")
    
    # 地面高度图用于指导地下生成
    surface_height = map_data.get_layer("height")
    
    # 为每一层生成内容
    for i in range(depth):
        layer_name = f"underground_{i}"
        layer = underground_layers[layer_name]
        
        # 生成基础地下地形
        logger.log(f"生成第{i+1}层地下地形...")
        if progress_callback:
            progress_callback(0.1 + (i / depth) * 0.3, f"生成第{i+1}层地下地形...")
        
        _generate_underground_terrain(
            layer["height"], 
            surface_height, 
            i, 
            depth, 
            map_data.get_layer("biome") if "biome" in map_data.layers else None,
            config,
            logger
        )
        
        # 生成矿物分布
        logger.log(f"生成第{i+1}层矿物分布...")
        if progress_callback:
            progress_callback(0.1 + (i / depth) * 0.3 + 0.1, f"生成第{i+1}层矿物分布...")
            
        _generate_minerals(
            map_data.mineral_layers[layer_name],
            layer["height"],
            i,
            depth,
            config,
            logger
        )
        
        # 生成地下内容
        logger.log(f"生成第{i+1}层地下内容...")
        if progress_callback:
            progress_callback(0.1 + (i / depth) * 0.3 + 0.2, f"生成第{i+1}层地下内容...")
            
        _generate_underground_content(
            layer["content"],
            layer["height"],
            surface_height,
            map_data.mineral_layers[layer_name],
            i,
            depth,
            config,
            logger
        )
    
    # 生成连接各层的洞穴网络
    logger.log("生成洞穴连通网络...")
    if progress_callback:
        progress_callback(0.6, "生成洞穴连通网络...")
        
    cave_networks = _generate_cave_networks(
        map_data,
        depth,
        map_data.get_layer("caves") if "caves" in map_data.layers else [],
        config,
        logger
    )
    
    # 添加所有洞穴网络
    for network in cave_networks:
        map_data.add_cave_network(network)
    
    # 生成地下水系统
    logger.log("生成地下水系统...")
    if progress_callback:
        progress_callback(0.8, "生成地下水系统...")
        
    _generate_underground_water_system(
        map_data,
        map_data.get_layer("rivers") if "rivers" in map_data.layers else None,
        config,
        logger
    )
    
    # 放置地下特殊结构
    logger.log("放置地下特殊结构...")
    if progress_callback:
        progress_callback(0.9, "放置地下特殊结构...")
        
    _place_underground_structures(
        map_data,
        config,
        logger
    )
    
    return map_data

def _generate_underground_terrain(height_map, surface_height, layer_index, total_layers, biome_map, config, logger):
    """生成地下地形基础层"""
    width, height = height_map.shape
    
    # 设置噪声种子
    seed = config.get("seed", random.randint(0, 999999))
    # 确保种子不为None
    if seed is None:
        seed = random.randint(0, 999999)
        logger.log(f"警告：配置中的种子为None，已自动生成随机种子：{seed}", "WARNING")
    
    noise_gen = OpenSimplex(seed=seed + layer_index * 1000)
    
    # 基于深度的参数调整
    scale = 0.01 * (1 + 0.5 * layer_index)  # 更深处尺度更大
    amplitude = 1.0 - 0.2 * layer_index  # 更深处变化更小
    
    # 生成基本噪声地形
    for y in range(height):
        for x in range(width):
            # 表面高度的影响，深度越深影响越小
            surface_influence = max(0, 1.0 - 0.3 * layer_index)
            surface_value = surface_height[y, x] * surface_influence
            
            # 噪声地形
            noise_value = noise_gen.noise2(x * scale, y * scale) * amplitude
            
            # 深度压力 - 更深处地形更平坦
            depth_pressure = 0.3 * layer_index
            
            # 组合各种影响
            terrain_value = (
                surface_value * 0.5 +  # 表面地形影响
                noise_value * (1.0 - depth_pressure) +  # 噪声贡献
                0.5  # 基础值
            )
            
            # 确保值在合理范围内
            height_map[y, x] = np.clip(terrain_value, 0.1, 0.9)
    
    # 根据生物群系添加特定特征
    if biome_map is not None:
        _add_biome_specific_underground_features(height_map, biome_map, layer_index, logger)
    
    return height_map

def _add_biome_specific_underground_features(height_map, biome_map, layer_index, logger):
    """根据地表生物群系添加特定的地下特征"""
    width, height = height_map.shape
    
    # 简单实现，为不同生物群系添加特征
    # 例如，山地下方可能有更多的洞穴和矿物，沙漠下方可能更平坦等
    try:
        for y in range(height):
            for x in range(width):
                biome_id = biome_map[y, x]
                
                # 基于生物群系ID调整地下特征
                if biome_id in [1, 2, 3]:  # 假设这些ID对应山地类型
                    # 山地下方更加崎岖
                    if random.random() < 0.2:
                        height_map[y, x] += random.uniform(-0.1, 0.15)
                elif biome_id in [4, 5]:  # 假设这些ID对应平原
                    # 平原下方更加平坦
                    height_map[y, x] = height_map[y, x] * 0.8 + 0.4 * 0.2
                elif biome_id in [8, 9]:  # 假设这些ID对应沙漠
                    # 沙漠下方分层明显
                    if random.random() < 0.3:
                        height_map[y, x] = round(height_map[y, x] * 4) / 4.0
    except Exception as e:
        logger.log(f"添加生物群系特定地下特征时出错: {e}", "WARNING")
    
    return height_map

def _generate_minerals(mineral_map, height_map, layer_index, total_layers, config, logger):
    """生成矿物分布"""
    width, height = mineral_map.shape
    
    # 设置噪声种子
    seed = config.get("seed", random.randint(0, 999999))
    # 确保种子不为None
    if seed is None:
        seed = random.randint(0, 999999)
    noise_gen_density = OpenSimplex(seed=seed + layer_index * 1000 + 1)
    noise_gen_type = OpenSimplex(seed=seed + layer_index * 1000 + 2)
    
    # 矿物类型和它们的稀有程度
    # 不同深度有不同的矿物分布规则
    mineral_distribution = [
        # Layer 0 (最浅层)
        {
            MineralType.NONE: 0.65,
            MineralType.COAL: 0.15,
            MineralType.IRON: 0.10,
            MineralType.COPPER: 0.08,
            MineralType.CRYSTAL: 0.02,
        },
        # Layer 1 (中层)
        {
            MineralType.NONE: 0.55,
            MineralType.COAL: 0.10,
            MineralType.IRON: 0.15,
            MineralType.COPPER: 0.10,
            MineralType.GOLD: 0.05,
            MineralType.SILVER: 0.03,
            MineralType.CRYSTAL: 0.02,
        },
        # Layer 2 (深层)
        {
            MineralType.NONE: 0.45,
            MineralType.IRON: 0.10,
            MineralType.GOLD: 0.08,
            MineralType.SILVER: 0.07,
            MineralType.DIAMOND: 0.05,
            MineralType.EMERALD: 0.05,
            MineralType.RUBY: 0.05,
            MineralType.SAPPHIRE: 0.05,
            MineralType.OBSIDIAN: 0.05,
            MineralType.CRYSTAL: 0.03,
            MineralType.MYTHRIL: 0.01,
            MineralType.ADAMANTINE: 0.01,
        },
    ]
    
    # 使用适合当前层的分布或最深层的分布
    distribution = mineral_distribution[min(layer_index, len(mineral_distribution)-1)]
    
    # 为每个位置生成矿物
    for y in range(height):
        for x in range(width):
            # 使用噪声函数创建矿物密度图
            density_scale = 0.03 * (1 + 0.5 * layer_index)
            density_value = (noise_gen_density.noise2(x * density_scale, y * density_scale) + 1) / 2
            
            # 地形影响 - 起伏区域可能有更多矿物
            terrain_factor = 1.0
            if x > 0 and y > 0 and x < width-1 and y < height-1:
                local_variance = np.std([
                    height_map[y-1, x], height_map[y+1, x],
                    height_map[y, x-1], height_map[y, x+1]
                ])
                terrain_factor = 1.0 + local_variance * 5.0
            
            # 最终密度由噪声和地形因素决定
            final_density = min(1.0, density_value * terrain_factor)
            
            # 如果密度足够高，生成矿物
            if final_density > 0.4:
                # 决定矿物类型
                type_scale = 0.02
                type_value = (noise_gen_type.noise2(x * type_scale, y * type_scale) + 1) / 2
                
                # 将噪声值映射到特定矿物
                cumulative_prob = 0
                mineral_type = MineralType.NONE
                
                for mineral, prob in distribution.items():
                    cumulative_prob += prob
                    if type_value <= cumulative_prob:
                        mineral_type = mineral
                        break
                
                mineral_map[y, x] = mineral_type
            else:
                mineral_map[y, x] = MineralType.NONE
    
    # 添加矿脉 - 大型集中区域
    num_veins = max(3, int((width * height) / 10000) * (layer_index + 1))
    for _ in range(num_veins):
        # 随机选择矿脉类型，排除NONE
        vein_minerals = [m for m in list(MineralType) if m != MineralType.NONE]
        vein_type = random.choice(vein_minerals)
        
        # 对于非常稀有的矿物，根据层深筛选
        if vein_type in [MineralType.DIAMOND, MineralType.EMERALD, MineralType.RUBY, 
                         MineralType.SAPPHIRE, MineralType.MYTHRIL, MineralType.ADAMANTINE]:
            # 最深层才有稀有矿物矿脉
            if layer_index < total_layers - 1:
                continue
        
        # 创建矿脉起点
        start_x = random.randint(0, width-1)
        start_y = random.randint(0, height-1)
        
        # 矿脉大小与稀有度成反比
        rarity_factor = 1.0
        if vein_type in [MineralType.DIAMOND, MineralType.EMERALD, MineralType.RUBY, 
                        MineralType.SAPPHIRE, MineralType.MYTHRIL, MineralType.ADAMANTINE]:
            rarity_factor = 0.5
        elif vein_type in [MineralType.GOLD, MineralType.SILVER]:
            rarity_factor = 0.7
        
        vein_size = int(random.randint(10, 40) * rarity_factor)
        
        # 使用简单的蔓延算法生成矿脉
        queue = [(start_x, start_y)]
        placed = set(queue)
        
        while queue and len(placed) < vein_size:
            x, y = queue.pop(0)
            
            if 0 <= x < width and 0 <= y < height:
                # 放置矿物
                mineral_map[y, x] = vein_type
                
                # 随机向四周蔓延
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in placed and 0 <= nx < width and 0 <= ny < height:
                        # 随机决定是否继续蔓延
                        if random.random() < 0.7:
                            queue.append((nx, ny))
                            placed.add((nx, ny))
    
    return mineral_map

def _generate_underground_content(content_map, height_map, surface_height, mineral_map, layer_index, total_layers, config, logger):
    """生成地下内容分布，如洞穴、隧道等"""
    width, height = content_map.shape
    
    # 设置噪声生成器
    seed = config.get("seed", random.randint(0, 999999))
    # 确保种子不为None
    if seed is None:
        seed = random.randint(0, 999999)
    noise_gen = OpenSimplex(seed=seed + layer_index * 1000 + 3)
    
    # 首先根据高度图初始化为土壤或岩石
    for y in range(height):
        for x in range(width):
            # 获取表面和地下高度
            surface_h = surface_height[y, x]
            underground_h = height_map[y, x]
            
            # 根据深度和高度决定基本地质类型
            if underground_h < 0.3:
                content_map[y, x] = UndergroundContentType.ROCK
            else:
                content_map[y, x] = UndergroundContentType.SOIL
    
    # 生成洞穴
    cave_scale = 0.03 * (1 + 0.2 * layer_index)  # 深层洞穴更大
    cave_threshold = 0.7 - 0.1 * layer_index  # 深层洞穴更多
    
    # 使用3D Perlin噪声来生成洞穴
    for y in range(height):
        for x in range(width):
            # 添加层索引作为z坐标，使每层洞穴分布不同
            noise_value = noise_gen.noise3(x * cave_scale, y * cave_scale, layer_index * 10)
            
            # 转换为0-1范围
            noise_value = (noise_value + 1) / 2
            
            # 如果噪声值高于阈值，形成洞穴
            if noise_value > cave_threshold:
                content_map[y, x] = UndergroundContentType.CAVE
    
    # 添加特殊洞穴 - 晶体洞穴、真菌洞穴等
    num_special_caves = max(2, int((width * height) / 20000))
    for _ in range(num_special_caves):
        # 随机选择特殊洞穴类型
        special_cave_types = [
            UndergroundContentType.CRYSTAL_CAVE,
            UndergroundContentType.FUNGAL_CAVE
        ]
        
        # 深层更可能有晶体洞穴
        weights = [1, 2] if layer_index == 0 else [2, 1]
        cave_type = random.choices(special_cave_types, weights=weights, k=1)[0]
        
        # 随机选择起点
        center_x = random.randint(0, width-1)
        center_y = random.randint(0, height-1)
        
        # 洞穴大小
        cave_radius = random.randint(5, 15)
        
        # 创建特殊洞穴
        for y in range(max(0, center_y - cave_radius), min(height, center_y + cave_radius + 1)):
            for x in range(max(0, center_x - cave_radius), min(width, center_x + cave_radius + 1)):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                # 中心区域是特殊洞穴，边缘是普通洞穴
                if distance < cave_radius * 0.7:
                    content_map[y, x] = cave_type
                elif distance < cave_radius:
                    content_map[y, x] = UndergroundContentType.CAVE
    
    # 添加地下结构 - 遗迹和废弃矿井
    if layer_index >= 1:  # 仅在更深层添加
        num_structures = max(1, int((width * height) / 50000))
        
        for _ in range(num_structures):
            # 随机选择结构类型
            structure_types = [
                UndergroundContentType.ANCIENT_RUINS,
                UndergroundContentType.ABANDONED_MINE
            ]
            structure_type = random.choice(structure_types)
            
            # 随机选择位置，避开现有洞穴
            attempts = 0
            while attempts < 100:
                center_x = random.randint(20, width-21)
                center_y = random.randint(20, height-21)
                
                # 检查位置是否适合
                suitable = True
                for check_y in range(center_y - 10, center_y + 11):
                    for check_x in range(center_x - 10, center_x + 11):
                        if 0 <= check_y < height and 0 <= check_x < width:
                            # 避开已有的洞穴和特殊内容
                            if content_map[check_y, check_x] not in [UndergroundContentType.SOIL, UndergroundContentType.ROCK]:
                                suitable = False
                                break
                    if not suitable:
                        break
                
                if suitable:
                    # 创建结构布局
                    # 此处使用简化方法，实际可以使用更复杂的结构生成算法
                    structure_size = random.randint(8, 15)
                    
                    if structure_type == UndergroundContentType.ANCIENT_RUINS:
                        # 创建一个矩形布局的遗迹
                        for sy in range(center_y - structure_size//2, center_y + structure_size//2 + 1):
                            for sx in range(center_x - structure_size//2, center_x + structure_size//2 + 1):
                                if 0 <= sy < height and 0 <= sx < width:
                                    # 边缘为墙壁，中间为空洞
                                    is_edge = (
                                        sy == center_y - structure_size//2 or
                                        sy == center_y + structure_size//2 or
                                        sx == center_x - structure_size//2 or
                                        sx == center_x + structure_size//2
                                    )
                                    
                                    if is_edge:
                                        content_map[sy, sx] = structure_type
                                    else:
                                        content_map[sy, sx] = UndergroundContentType.CAVE
                    
                    elif structure_type == UndergroundContentType.ABANDONED_MINE:
                        # 创建连通的隧道网络
                        tunnels = []
                        start_points = [
                            (center_x, center_y),
                            (center_x + structure_size//2, center_y),
                            (center_x - structure_size//2, center_y),
                            (center_x, center_y + structure_size//2),
                            (center_x, center_y - structure_size//2)
                        ]
                        
                        for start_x, start_y in start_points:
                            if 0 <= start_y < height and 0 <= start_x < width:
                                # 随机方向的隧道
                                angle = random.uniform(0, 2 * np.pi)
                                tunnel_length = random.randint(5, 12)
                                
                                tunnel = []
                                curr_x, curr_y = start_x, start_y
                                
                                for _ in range(tunnel_length):
                                    tunnel.append((curr_x, curr_y))
                                    
                                    # 移动到下一点，略微随机改变方向
                                    angle += random.uniform(-0.5, 0.5)
                                    next_x = int(curr_x + np.cos(angle))
                                    next_y = int(curr_y + np.sin(angle))
                                    
                                    # 确保在地图范围内
                                    if 0 <= next_y < height and 0 <= next_x < width:
                                        curr_x, curr_y = next_x, next_y
                                    else:
                                        break
                                
                                tunnels.append(tunnel)
                        
                        # 放置隧道
                        for tunnel in tunnels:
                            for tx, ty in tunnel:
                                if 0 <= ty < height and 0 <= tx < width:
                                    content_map[ty, tx] = structure_type
                    
                    break
                
                attempts += 1
    
    return content_map

def _generate_cave_networks(map_data, depth, surface_cave_entrances, config, logger):
    """生成连接各地下层的洞穴网络系统"""
    width = map_data.width
    height = map_data.height
    
    cave_networks = []
    
    # 生成从地表洞穴入口延伸的洞穴网络
    for entrance in surface_cave_entrances:
        if isinstance(entrance, dict) and "x" in entrance and "y" in entrance:
            entrance_x, entrance_y = entrance["x"], entrance["y"]
        elif isinstance(entrance, (list, tuple)) and len(entrance) >= 2:
            entrance_x, entrance_y = entrance[0], entrance[1]
        else:
            continue  # 跳过无效的入口
        
        # 随机决定这个洞穴网络能达到的最大深度
        max_depth = random.randint(1, depth)
        
        # 创建洞穴网络
        network = {
            "id": len(cave_networks),
            "entrance": (entrance_x, entrance_y),
            "paths": [],
            "chambers": [],
            "max_depth": max_depth
        }
        
        # 生成主路径 - 从入口到最大深度
        main_path = _generate_cave_path(
            (entrance_x, entrance_y), 
            max_depth, 
            width, 
            height, 
            config
        )
        network["paths"].append(main_path)
        
        # 为每层添加岔路和洞穴房间
        for layer_idx in range(max_depth):
            # 找到这层的点
            layer_points = [p for p in main_path if p[2] == layer_idx]
            
            if layer_points:
                # 随机选择分支点
                branch_point = random.choice(layer_points)
                
                # 生成1-3条岔路
                num_branches = random.randint(1, 3)
                
                for _ in range(num_branches):
                    # 岔路最多到下一层
                    branch_max_depth = min(layer_idx + 1, max_depth - 1)
                    
                    branch_path = _generate_cave_path(
                        (branch_point[0], branch_point[1], branch_point[2]),
                        branch_max_depth, 
                        width, 
                        height,
                        config,
                        is_branch=True
                    )
                    
                    if branch_path:
                        network["paths"].append(branch_path)
                
                # 添加1-2个洞穴房间
                num_chambers = random.randint(1, 2)
                
                for _ in range(num_chambers):
                    chamber_point = random.choice(layer_points)
                    
                    # 房间大小随深度增加
                    chamber_size = random.randint(3, 6 + layer_idx)
                    
                    chamber = {
                        "center": chamber_point,
                        "size": chamber_size,
                        "type": random.choice([
                            "regular", "crystal", "fungal", "water",
                            "lava" if layer_idx >= depth - 1 else "regular"
                        ])
                    }
                    
                    network["chambers"].append(chamber)
        
        cave_networks.append(network)
    
    # 应用洞穴网络到地下层
    for network in cave_networks:
        # 处理所有路径
        for path in network["paths"]:
            for x, y, layer_idx in path:
                if 0 <= layer_idx < depth:
                    layer_name = f"underground_{layer_idx}"
                    if layer_name in map_data.underground_layers:
                        content_layer = map_data.underground_layers[layer_name]["content"]
                        # 确保在数组范围内
                        if 0 <= y < content_layer.shape[0] and 0 <= x < content_layer.shape[1]:
                            content_layer[y, x] = UndergroundContentType.TUNNEL
        
        # 处理所有房间
        for chamber in network["chambers"]:
            cx, cy, layer_idx = chamber["center"]
            size = chamber["size"]
            chamber_type = chamber["type"]
            
            if 0 <= layer_idx < depth:
                layer_name = f"underground_{layer_idx}"
                if layer_name in map_data.underground_layers:
                    content_layer = map_data.underground_layers[layer_name]["content"]
                    
                    # 确定房间内容类型
                    content_type = UndergroundContentType.CAVE
                    if chamber_type == "crystal":
                        content_type = UndergroundContentType.CRYSTAL_CAVE
                    elif chamber_type == "fungal":
                        content_type = UndergroundContentType.FUNGAL_CAVE
                    elif chamber_type == "water":
                        content_type = UndergroundContentType.WATER
                    elif chamber_type == "lava":
                        content_type = UndergroundContentType.LAVA
                    
                    # 创建圆形房间
                    for y in range(max(0, cy - size), min(content_layer.shape[0], cy + size + 1)):
                        for x in range(max(0, cx - size), min(content_layer.shape[1], cx + size + 1)):
                            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                            if dist <= size:
                                content_layer[y, x] = content_type
    
    return cave_networks

def _generate_cave_path(start, max_depth, width, height, config, is_branch=False):
    """生成从起点到指定深度的随机洞穴路径"""
    # 起点可能是二元或三元组
    if len(start) == 2:
        start_x, start_y = start
        curr_depth = 0
    else:
        start_x, start_y, curr_depth = start
    
    path = [(start_x, start_y, curr_depth)]
    
    # 如果已达到最大深度，直接返回
    if curr_depth >= max_depth:
        return path
    
    # 生成路径参数
    step_count = random.randint(5, 15) if not is_branch else random.randint(3, 8)
    descend_probability = 0.3  # 每步有30%的概率下降到下一层
    
    curr_x, curr_y = start_x, start_y
    
    for _ in range(step_count):
        # 随机方向
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        
        # 确保移动有效
        if dx == 0 and dy == 0:
            dx = random.choice([-1, 1])
        
        # 新位置
        new_x = max(0, min(width - 1, curr_x + dx))
        new_y = max(0, min(height - 1, curr_y + dy))
        
        # 随机决定是否下降一层
        if random.random() < descend_probability and curr_depth < max_depth:
            curr_depth += 1
        
        # 添加到路径
        path.append((new_x, new_y, curr_depth))
        curr_x, curr_y = new_x, new_y
    
    return path

def _generate_underground_water_system(map_data, surface_rivers, config, logger):
    """生成地下水系统，包括地下河流、地下湖和渗水区"""
    depth = map_data.underground_depth
    width = map_data.width
    height = map_data.height
    
    # 解析配置
    seed = config.get("seed", random.randint(0, 999999))
    # 确保种子不为None
    if seed is None:
        seed = random.randint(0, 999999)
    water_prevalence = config.get("underground_water_prevalence", 0.5)  # 0-1范围
    
    # 初始化地下水系统
    map_data.underground_water = {
        "rivers": [],
        "lakes": [],
        "seepage_areas": []
    }
    
    # 从表面河流确定地下水入口点
    seepage_points = []
    
    if surface_rivers is not None:
        # 每隔一定距离采样河流点作为渗水起点
        sample_interval = max(10, int(width * height / 2000))
        
        for y in range(0, height, sample_interval):
            for x in range(0, width, sample_interval):
                if 0 <= y < surface_rivers.shape[0] and 0 <= x < surface_rivers.shape[1]:
                    # 检查周围的河流点
                    found_river = False
                    for check_y in range(max(0, y-3), min(surface_rivers.shape[0], y+4)):
                        for check_x in range(max(0, x-3), min(surface_rivers.shape[1], x+4)):
                            if surface_rivers[check_y, check_x]:
                                seepage_points.append((check_x, check_y))
                                found_river = True
                                break
                        if found_river:
                            break
    
    # 添加一些随机渗水点
    num_random_points = max(5, int(water_prevalence * width * height / 10000))
    for _ in range(num_random_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        seepage_points.append((x, y))
    
    # 对每个渗水点，生成地下水流
    for start_x, start_y in seepage_points:
        # 决定水能够渗透的最大深度
        max_water_depth = random.randint(0, depth - 1)
        
        # 创建主渗水区
        for layer_idx in range(max_water_depth + 1):
            layer_name = f"underground_{layer_idx}"
            if layer_name in map_data.underground_layers:
                content_layer = map_data.underground_layers[layer_name]["content"]
                
                # 渗水区半径随深度减小
                seepage_radius = max(2, 8 - layer_idx * 2)
                
                # 创建渗水区
                for y in range(max(0, start_y - seepage_radius), min(height, start_y + seepage_radius + 1)):
                    for x in range(max(0, start_x - seepage_radius), min(width, start_x + seepage_radius + 1)):
                        dist = ((x - start_x) ** 2 + (y - start_y) ** 2) ** 0.5
                        
                        if dist <= seepage_radius:
                            # 只在洞穴和隧道中添加水
                            if content_layer[y, x] in [UndergroundContentType.CAVE, UndergroundContentType.TUNNEL]:
                                content_layer[y, x] = UndergroundContentType.WATER
                
                # 记录渗水区
                map_data.underground_water["seepage_areas"].append({
                    "center": (start_x, start_y),
                    "radius": seepage_radius,
                    "layer": layer_idx
                })
        
        # 随机决定是否形成地下河流
        if random.random() < 0.3 * water_prevalence:
            # 选择一个中间层
            river_layer = random.randint(0, min(max_water_depth, depth - 1))
            layer_name = f"underground_{river_layer}"
            
            if layer_name in map_data.underground_layers:
                content_layer = map_data.underground_layers[layer_name]["content"]
                height_layer = map_data.underground_layers[layer_name]["height"]
                
                # 生成河流路径
                river_points = _generate_underground_river_path(
                    start_x, start_y, width, height, height_layer, content_layer,
                    min_length=10, max_length=30, seed=seed + len(map_data.underground_water["rivers"])
                )
                
                # 将路径应用到内容层
                for rx, ry in river_points:
                    if 0 <= ry < content_layer.shape[0] and 0 <= rx < content_layer.shape[1]:
                        content_layer[ry, rx] = UndergroundContentType.UNDERGROUND_RIVER
                
                # 记录河流
                map_data.underground_water["rivers"].append({
                    "path": river_points,
                    "layer": river_layer
                })
    
    # 在深层添加随机地下湖
    for layer_idx in range(depth):
        layer_name = f"underground_{layer_idx}"
        if layer_name in map_data.underground_layers:
            content_layer = map_data.underground_layers[layer_name]["content"]
            
            # 深层更可能有湖
            lake_chance = 0.2 + 0.1 * layer_idx
            if random.random() < lake_chance * water_prevalence:
                # 随机选择湖泊中心
                lake_x = random.randint(20, width - 21)
                lake_y = random.randint(20, height - 21)
                
                # 湖泊大小
                lake_size = random.randint(5, 15)
                
                # 湖泊类型 - 最深层可能有岩浆湖
                lake_type = UndergroundContentType.WATER
                if layer_idx == depth - 1 and random.random() < 0.3:
                    lake_type = UndergroundContentType.LAVA
                
                # 创建湖泊
                lake_points = []
                for y in range(max(0, lake_y - lake_size), min(height, lake_y + lake_size + 1)):
                    for x in range(max(0, lake_x - lake_size), min(width, lake_x + lake_size + 1)):
                        dist = ((x - lake_x) ** 2 + (y - lake_y) ** 2) ** 0.5
                        
                        # 使用噪声创建自然的湖岸线
                        noise_val = OpenSimplex(seed=seed + layer_idx * 1000).noise2(x * 0.1, y * 0.1) * 2
                        
                        if dist <= lake_size + noise_val:
                            # 只在洞穴区域创建湖
                            if content_layer[y, x] in [UndergroundContentType.CAVE, UndergroundContentType.TUNNEL]:
                                content_layer[y, x] = lake_type
                                lake_points.append((x, y))
                
                if lake_points:
                    # 记录湖泊
                    map_data.underground_water["lakes"].append({
                        "center": (lake_x, lake_y),
                        "points": lake_points,
                        "type": "lava" if lake_type == UndergroundContentType.LAVA else "water",
                        "layer": layer_idx
                    })
    
    return map_data.underground_water

def _generate_underground_river_path(start_x, start_y, width, height, height_map, content_map, min_length=10, max_length=30, seed=None):
    """生成地下河流路径"""
    if seed is not None:
        random.seed(seed)
    
    path = [(start_x, start_y)]
    curr_x, curr_y = start_x, start_y
    
    # 河流长度
    river_length = random.randint(min_length, max_length)
    
    # 生成河流路径 - 总是向低处流动
    for _ in range(river_length):
        # 检查周围8个方向
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        valid_moves = []
        
        for dx, dy in directions:
            nx, ny = curr_x + dx, curr_y + dy
            
            # 检查是否在地图范围内
            if 0 <= ny < height and 0 <= nx < width:
                # 检查是否已经是河流的一部分
                if (nx, ny) not in path:
                    # 检查是否在洞穴/隧道中
                    if content_map[ny, nx] in [UndergroundContentType.CAVE, UndergroundContentType.TUNNEL]:
                        # 添加到有效移动
                        valid_moves.append((nx, ny, height_map[ny, nx]))
        
        if not valid_moves:
            break
        
        # 优先选择高度较低的方向，使河流自然流向低处
        valid_moves.sort(key=lambda m: m[2])
        
        # 随机选择一个较低点，加入一些随机性
        next_point = random.choice(valid_moves[:max(1, len(valid_moves)//2)])
        curr_x, curr_y = next_point[0], next_point[1]
        path.append((curr_x, curr_y))
    
    # 重置随机数种子
    if seed is not None:
        random.seed()
    
    return path

def _place_underground_structures(map_data, config, logger):
    """放置地下特殊结构，如神庙、废弃实验室等"""
    depth = map_data.underground_depth
    width = map_data.width
    height = map_data.height
    
    # 解析配置
    seed = config.get("seed", random.randint(0, 999999))
    # 确保种子不为None
    if seed is None:
        seed = random.randint(0, 999999)
    structure_density = config.get("underground_structure_density", 0.5)  # 0-1范围
    
    # 定义可能的结构类型
    structure_types = [
        "ancient_temple",
        "abandoned_laboratory",
        "crystal_formation",
        "mushroom_grove",
        "lava_forge",
        "underground_settlement",
        "treasure_vault"
    ]
    
    # 为每一层放置结构
    for layer_idx in range(depth):
        layer_name = f"underground_{layer_idx}"
        if layer_name in map_data.underground_layers:
            content_layer = map_data.underground_layers[layer_name]["content"]
            
            # 每层结构数量
            num_structures = max(1, int(width * height * structure_density / 50000))
            
            # 深层更多的结构
            num_structures += layer_idx
            
            # 为每个结构找合适的位置
            structures_placed = 0
            
            for _ in range(num_structures * 3):  # 尝试次数是结构数的3倍
                if structures_placed >= num_structures:
                    break
                
                # 随机位置
                struct_x = random.randint(20, width - 21)
                struct_y = random.randint(20, height - 21)
                
                # 检查周围是否有足够的洞穴空间
                has_space = True
                cave_count = 0
                
                for check_y in range(struct_y - 10, struct_y + 11):
                    for check_x in range(struct_x - 10, struct_x + 11):
                        if 0 <= check_y < height and 0 <= check_x < width:
                            if content_layer[check_y, check_x] == UndergroundContentType.CAVE:
                                cave_count += 1
                            elif content_layer[check_y, check_x] not in [UndergroundContentType.SOIL, UndergroundContentType.ROCK]:
                                # 如果有其他结构，不放置
                                has_space = False
                                break
                    if not has_space:
                        break
                
                # 需要有足够的洞穴空间
                if has_space and cave_count >= 20:
                    # 随机选择结构类型，但某些类型只在特定层出现
                    available_types = structure_types.copy()
                    
                    if layer_idx == 0:  # 最浅层
                        if "lava_forge" in available_types:
                            available_types.remove("lava_forge")
                    
                    if layer_idx < depth - 1:  # 非最深层
                        # 一些稀有结构只在最深层
                        rare_deep_structures = ["ancient_temple", "treasure_vault"]
                        for struct in rare_deep_structures:
                            if random.random() < 0.7 and struct in available_types:  # 70%的几率从非最深层移除
                                available_types.remove(struct)
                    
                    # 选择类型
                    structure_type = random.choice(available_types)
                    
                    # 创建结构
                    structure_data = _create_underground_structure(
                        structure_type, struct_x, struct_y, layer_idx, seed + structures_placed
                    )
                    
                    # 应用结构到地图
                    for point_x, point_y in structure_data["points"]:
                        if 0 <= point_y < height and 0 <= point_x < width:
                            content_layer[point_y, point_x] = UndergroundContentType.ANCIENT_RUINS
                    
                    # 添加到结构列表
                    map_data.add_underground_structure(layer_idx, structure_data)
                    
                    structures_placed += 1
    
    return map_data

def _create_underground_structure(structure_type, center_x, center_y, layer, seed=None):
    """创建地下结构数据"""
    if seed is not None:
        random.seed(seed)
    
    structure = {
        "type": structure_type,
        "center": (center_x, center_y),
        "layer": layer,
        "points": [],
        "features": []
    }
    
    # 根据结构类型创建不同的布局
    if structure_type == "ancient_temple":
        # 创建矩形神殿
        temple_width = random.randint(7, 12)
        temple_height = random.randint(7, 12)
        
        # 外墙
        for x in range(center_x - temple_width//2, center_x + temple_width//2 + 1):
            for y in range(center_y - temple_height//2, center_y + temple_height//2 + 1):
                is_wall = (
                    x == center_x - temple_width//2 or
                    x == center_x + temple_width//2 or
                    y == center_y - temple_height//2 or
                    y == center_y + temple_height//2
                )
                
                if is_wall:
                    structure["points"].append((x, y))
        
        # 添加入口
        entrance_side = random.choice(["north", "south", "east", "west"])
        
        if entrance_side == "north":
            entrance_x = center_x
            entrance_y = center_y - temple_height//2
        elif entrance_side == "south":
            entrance_x = center_x
            entrance_y = center_y + temple_height//2
        elif entrance_side == "east":
            entrance_x = center_x + temple_width//2
            entrance_y = center_y
        else:  # west
            entrance_x = center_x - temple_width//2
            entrance_y = center_y
        
        # 移除入口点
        if (entrance_x, entrance_y) in structure["points"]:
            structure["points"].remove((entrance_x, entrance_y))
        
        # 添加内部特征
        structure["features"].append({
            "type": "altar",
            "position": (center_x, center_y)
        })
        
        # 添加随机支柱
        num_pillars = random.randint(2, 4)
        pillar_positions = []
        
        for _ in range(num_pillars):
            px = random.randint(center_x - temple_width//2 + 2, center_x + temple_width//2 - 2)
            py = random.randint(center_y - temple_height//2 + 2, center_y + temple_height//2 - 2)
            pillar_positions.append((px, py))
            structure["points"].append((px, py))
        
        structure["features"].append({
            "type": "pillars",
            "positions": pillar_positions
        })
    
    elif structure_type == "abandoned_laboratory":
        # 创建不规则的房间网络
        rooms = []
        
        # 主房间
        main_room_width = random.randint(5, 8)
        main_room_height = random.randint(5, 8)
        
        main_room = {
            "x1": center_x - main_room_width//2,
            "y1": center_y - main_room_height//2,
            "x2": center_x + main_room_width//2,
            "y2": center_y + main_room_height//2
        }
        
        rooms.append(main_room)
        
        # 添加2-4个次要房间
        num_side_rooms = random.randint(2, 4)
        
        for _ in range(num_side_rooms):
            # 随机选择主房间的一侧
            side = random.choice(["north", "south", "east", "west"])
            
            if side == "north":
                room_x1 = random.randint(main_room["x1"], main_room["x2"] - 3)
                room_x2 = room_x1 + random.randint(3, 5)
                room_y1 = main_room["y1"] - random.randint(3, 5)
                room_y2 = main_room["y1"]
            elif side == "south":
                room_x1 = random.randint(main_room["x1"], main_room["x2"] - 3)
                room_x2 = room_x1 + random.randint(3, 5)
                room_y1 = main_room["y2"]
                room_y2 = main_room["y2"] + random.randint(3, 5)
            elif side == "east":
                room_x1 = main_room["x2"]
                room_x2 = main_room["x2"] + random.randint(3, 5)
                room_y1 = random.randint(main_room["y1"], main_room["y2"] - 3)
                room_y2 = room_y1 + random.randint(3, 5)
            else:  # west
                room_x1 = main_room["x1"] - random.randint(3, 5)
                room_x2 = main_room["x1"]
                room_y1 = random.randint(main_room["y1"], main_room["y2"] - 3)
                room_y2 = room_y1 + random.randint(3, 5)
            
            rooms.append({
                "x1": room_x1,
                "y1": room_y1,
                "x2": room_x2,
                "y2": room_y2
            })
        
        # 创建房间墙壁
        for room in rooms:
            # 上下墙
            for x in range(room["x1"], room["x2"] + 1):
                structure["points"].append((x, room["y1"]))
                structure["points"].append((x, room["y2"]))
            
            # 左右墙
            for y in range(room["y1"] + 1, room["y2"]):
                structure["points"].append((room["x1"], y))
                structure["points"].append((room["x2"], y))
        
        # 添加房间之间的门
        for side_room in rooms[1:]:
            # 确定和主房间相邻的墙
            if side_room["x2"] == main_room["x1"]:  # 西侧房间
                door_y = random.randint(max(side_room["y1"] + 1, main_room["y1"] + 1),
                                      min(side_room["y2"] - 1, main_room["y2"] - 1))
                door_x = side_room["x2"]
                if (door_x, door_y) in structure["points"]:
                    structure["points"].remove((door_x, door_y))
            elif side_room["x1"] == main_room["x2"]:  # 东侧房间
                door_y = random.randint(max(side_room["y1"] + 1, main_room["y1"] + 1),
                                      min(side_room["y2"] - 1, main_room["y2"] - 1))
                door_x = side_room["x1"]
                if (door_x, door_y) in structure["points"]:
                    structure["points"].remove((door_x, door_y))
            elif side_room["y2"] == main_room["y1"]:  # 北侧房间
                door_x = random.randint(max(side_room["x1"] + 1, main_room["x1"] + 1),
                                      min(side_room["x2"] - 1, main_room["x2"] - 1))
                door_y = side_room["y2"]
                if (door_x, door_y) in structure["points"]:
                    structure["points"].remove((door_x, door_y))
            elif side_room["y1"] == main_room["y2"]:  # 南侧房间
                door_x = random.randint(max(side_room["x1"] + 1, main_room["x1"] + 1),
                                      min(side_room["x2"] - 1, main_room["x2"] - 1))
                door_y = side_room["y1"]
                if (door_x, door_y) in structure["points"]:
                    structure["points"].remove((door_x, door_y))
        
        # 添加外部入口
        entrance_room = random.choice(rooms)
        entrance_side = random.choice(["north", "south", "east", "west"])
        
        if entrance_side == "north" and entrance_room["y1"] > 1:
            entrance_x = random.randint(entrance_room["x1"] + 1, entrance_room["x2"] - 1)
            entrance_y = entrance_room["y1"]
        elif entrance_side == "south":
            entrance_x = random.randint(entrance_room["x1"] + 1, entrance_room["x2"] - 1)
            entrance_y = entrance_room["y2"]
        elif entrance_side == "east":
            entrance_x = entrance_room["x2"]
            entrance_y = random.randint(entrance_room["y1"] + 1, entrance_room["y2"] - 1)
        else:  # west
            entrance_x = entrance_room["x1"]
            entrance_y = random.randint(entrance_room["y1"] + 1, entrance_room["y2"] - 1)
        
        # 移除入口点
        if (entrance_x, entrance_y) in structure["points"]:
            structure["points"].remove((entrance_x, entrance_y))
        
        # 添加一些实验室特征
        structure["features"].append({
            "type": "equipment",
            "positions": [
                (random.randint(main_room["x1"] + 1, main_room["x2"] - 1), 
                 random.randint(main_room["y1"] + 1, main_room["y2"] - 1))
                for _ in range(random.randint(1, 3))
            ]
        })
        
        # 在随机房间中添加一个特殊实验装置
        special_room = random.choice(rooms)
        structure["features"].append({
            "type": "special_device",
            "position": (
                (special_room["x1"] + special_room["x2"]) // 2,
                (special_room["y1"] + special_room["y2"]) // 2
            )
        })
    
    elif structure_type == "crystal_formation":
        # 创建晶体簇
        formation_radius = random.randint(4, 7)
        
        # 主晶体簇
        for angle in range(0, 360, random.randint(20, 40)):
            # 随机晶体长度
            length = random.randint(2, formation_radius)
            rad_angle = np.radians(angle)
            
            # 晶体从中心向外延伸
            for r in range(length):
                x = center_x + int(r * np.cos(rad_angle))
                y = center_y + int(r * np.sin(rad_angle))
                structure["points"].append((x, y))
        
        # 添加一些随机小晶体
        num_small_crystals = random.randint(3, 8)
        for _ in range(num_small_crystals):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.randint(formation_radius + 1, formation_radius + 4)
            
            crystal_x = center_x + int(dist * np.cos(angle))
            crystal_y = center_y + int(dist * np.sin(angle))
            
            # 添加小晶体
            structure["points"].append((crystal_x, crystal_y))
            
            # 有时添加更小的晶体
            if random.random() < 0.5:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if random.random() < 0.3:
                        structure["points"].append((crystal_x + dx, crystal_y + dy))
        
        # 添加特殊晶体特征
        structure["features"].append({
            "type": "glowing_crystal",
            "position": (center_x, center_y)
        })
    
    elif structure_type == "mushroom_grove":
        # 创建蘑菇林
        grove_radius = random.randint(6, 10)
        
        # 在半径内随机放置蘑菇
        num_mushrooms = random.randint(10, 20)
        mushroom_positions = []
        
        for _ in range(num_mushrooms):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(1, grove_radius)
            
            x = center_x + int(dist * np.cos(angle))
            y = center_y + int(dist * np.sin(angle))
            
            mushroom_positions.append((x, y))
            structure["points"].append((x, y))
        
        # 添加特殊大型蘑菇
        structure["features"].append({
            "type": "giant_mushroom",
            "position": (center_x, center_y)
        })
        
        # 添加蘑菇圈
        if random.random() < 0.7:
            ring_radius = random.randint(grove_radius // 2, grove_radius - 1)
            for angle in range(0, 360, random.randint(15, 30)):
                rad_angle = np.radians(angle)
                x = center_x + int(ring_radius * np.cos(rad_angle))
                y = center_y + int(ring_radius * np.sin(rad_angle))
                structure["points"].append((x, y))
                
                if (x, y) not in mushroom_positions:
                    mushroom_positions.append((x, y))
        
        structure["features"].append({
            "type": "mushrooms",
            "positions": mushroom_positions
        })
    
    elif structure_type == "lava_forge":
        # 创建熔岩锻造室
        forge_width = random.randint(7, 10)
        forge_height = random.randint(7, 10)
        
        # 石墙
        for x in range(center_x - forge_width//2, center_x + forge_width//2 + 1):
            for y in range(center_y - forge_height//2, center_y + forge_height//2 + 1):
                is_wall = (
                    x == center_x - forge_width//2 or
                    x == center_x + forge_width//2 or
                    y == center_y - forge_height//2 or
                    y == center_y + forge_height//2
                )
                
                if is_wall:
                    structure["points"].append((x, y))
        
        # 添加入口
        entrance_x = center_x
        entrance_y = center_y - forge_height//2
        if (entrance_x, entrance_y) in structure["points"]:
            structure["points"].remove((entrance_x, entrance_y))
        
        # 添加熔岩池
        lava_positions = []
        lava_radius = min(forge_width, forge_height) // 3
        
        for y in range(center_y - lava_radius, center_y + lava_radius + 1):
            for x in range(center_x - lava_radius, center_x + lava_radius + 1):
                if ((x - center_x) ** 2 + (y - center_y) ** 2) <= lava_radius ** 2:
                    lava_positions.append((x, y))
        
        structure["features"].append({
            "type": "lava_pool",
            "positions": lava_positions
        })
        
        # 添加锻造设备
        anvil_x = center_x
        anvil_y = center_y - lava_radius - 1
        
        structure["features"].append({
            "type": "anvil",
            "position": (anvil_x, anvil_y)
        })
        
        # 添加工匠工作台
        for i in range(2):
            table_x = center_x - forge_width//2 + 2 + i * (forge_width - 4)
            table_y = center_y + forge_height//2 - 2
            
            structure["features"].append({
                "type": "workbench",
                "position": (table_x, table_y)
            })
    
    elif structure_type == "underground_settlement":
        # 创建地下定居点
        village_radius = random.randint(8, 12)
        
        # 创建中央广场
        plaza_radius = village_radius // 3
        
        # 添加环形围墙
        for angle in range(0, 360, 5):
            rad_angle = np.radians(angle)
            x = center_x + int(village_radius * np.cos(rad_angle))
            y = center_y + int(village_radius * np.sin(rad_angle))
            structure["points"].append((x, y))
        
        # 添加入口通道
        entrance_angle = random.uniform(0, 2 * np.pi)
        for r in range(village_radius - 1, village_radius + 3):
            x = center_x + int(r * np.cos(entrance_angle))
            y = center_y + int(r * np.sin(entrance_angle))
            if (x, y) in structure["points"]:
                structure["points"].remove((x, y))
        
        # 添加房屋
        num_houses = random.randint(4, 8)
        houses = []
        
        for i in range(num_houses):
            house_angle = 2 * np.pi * i / num_houses
            house_dist = random.uniform(plaza_radius + 1, village_radius - 2)
            
            house_x = center_x + int(house_dist * np.cos(house_angle))
            house_y = center_y + int(house_dist * np.sin(house_angle))
            
            house_size = random.randint(2, 3)
            house = {
                "center": (house_x, house_y),
                "size": house_size
            }
            houses.append(house)
            
            # 创建房屋围墙
            for dx in range(-house_size, house_size + 1):
                for dy in range(-house_size, house_size + 1):
                    if abs(dx) == house_size or abs(dy) == house_size:
                        structure["points"].append((house_x + dx, house_y + dy))
            
            # 添加门
            door_angle = house_angle + np.pi  # 朝向中心
            door_x = house_x + int(house_size * np.cos(door_angle))
            door_y = house_y + int(house_size * np.sin(door_angle))
            
            if (door_x, door_y) in structure["points"]:
                structure["points"].remove((door_x, door_y))
        
        structure["features"].append({
            "type": "settlement",
            "houses": houses,
            "central_plaza": (center_x, center_y)
        })
        
        # 添加中央特征
        central_feature = random.choice(["well", "statue", "market", "shrine"])
        structure["features"].append({
            "type": central_feature,
            "position": (center_x, center_y)
        })
    
    elif structure_type == "treasure_vault":
        # 创建宝藏库
        vault_size = random.randint(5, 8)
        
        # 创建圆形或方形宝库
        vault_shape = random.choice(["circle", "square"])
        
        if vault_shape == "circle":
            # 圆形宝库
            for angle in range(0, 360, 5):
                rad_angle = np.radians(angle)
                x = center_x + int(vault_size * np.cos(rad_angle))
                y = center_y + int(vault_size * np.sin(rad_angle))
                structure["points"].append((x, y))
        else:
            # 方形宝库
            for x in range(center_x - vault_size, center_x + vault_size + 1):
                for y in range(center_y - vault_size, center_y + vault_size + 1):
                    is_wall = (
                        x == center_x - vault_size or
                        x == center_x + vault_size or
                        y == center_y - vault_size or
                        y == center_y + vault_size
                    )
                    
                    if is_wall:
                        structure["points"].append((x, y))
        
        # 添加入口和通道
        entrance_angle = random.uniform(0, 2 * np.pi)
        entrance_length = random.randint(3, 6)
        
        for r in range(vault_size, vault_size + entrance_length + 1):
            x = center_x + int(r * np.cos(entrance_angle))
            y = center_y + int(r * np.sin(entrance_angle))
            
            # 墙壁
            for offset in [-1, 1]:
                perp_angle = entrance_angle + np.pi/2
                wall_x = x + int(offset * np.cos(perp_angle))
                wall_y = y + int(offset * np.sin(perp_angle))
                structure["points"].append((wall_x, wall_y))
            
            # 通道本身保持空白
            if (x, y) in structure["points"]:
                structure["points"].remove((x, y))
        
        # 添加宝藏
        treasure_positions = []
        
        # 中央主宝藏
        structure["features"].append({
            "type": "main_treasure",
            "position": (center_x, center_y)
        })
        
        # 随机分布的小宝藏
        for _ in range(random.randint(3, 7)):
            if vault_shape == "circle":
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(1, vault_size * 0.7)
                tx = center_x + int(dist * np.cos(angle))
                ty = center_y + int(dist * np.sin(angle))
            else:
                tx = random.randint(center_x - vault_size + 2, center_x + vault_size - 2)
                ty = random.randint(center_y - vault_size + 2, center_y + vault_size - 2)
            
            treasure_positions.append((tx, ty))
        
        structure["features"].append({
            "type": "treasures",
            "positions": treasure_positions
        })
        
        # 随机添加陷阱
        if random.random() < 0.7:
            trap_positions = []
            num_traps = random.randint(2, 5)
            
            # 陷阱通常放在通向宝藏的路上
            for _ in range(num_traps):
                if vault_shape == "circle":
                    angle = entrance_angle + random.uniform(-np.pi/4, np.pi/4)
                    dist = random.uniform(vault_size * 0.3, vault_size * 0.9)
                    tx = center_x + int(dist * np.cos(angle))
                    ty = center_y + int(dist * np.sin(angle))
                else:
                    # 随机选择放在入口和中心之间的位置
                    entrance_x = center_x + int(vault_size * np.cos(entrance_angle))
                    entrance_y = center_y + int(vault_size * np.sin(entrance_angle))
                    
                    t = random.uniform(0.3, 0.8)
                    tx = int(entrance_x * (1 - t) + center_x * t)
                    ty = int(entrance_y * (1 - t) + center_y * t)
                
                trap_positions.append((tx, ty))
            
            structure["features"].append({
                "type": "traps",
                "positions": trap_positions
            })
    
    # 确保所有点是唯一的
    structure["points"] = list(set(structure["points"]))
    
    # 重置随机数种子
    if seed is not None:
        random.seed()
    
    return structure

def integrate_underground_to_map_data(map_data, config, logger, progress_callback=None):
    """将地下系统集成到地图数据中
    
    Args:
        map_data: MapData实例
        config: 配置参数
        logger: 日志记录器
        progress_callback: 进度回调函数，接受进度值(0-1)和消息
        
    Returns:
        更新后的map_data
    """
    # 检查是否启用地下功能
    if not config.get("enable_underground", False):
        logger.log("地下功能未启用，跳过地下生成")
        if progress_callback:
            progress_callback(1.0, "地下功能未启用，已跳过")
        return map_data
    
    # 生成地下层
    logger.log("开始生成地下系统")
    
    try:
        # 调用进度回调，表示开始生成
        if progress_callback:
            progress_callback(0.1, "准备地下系统数据...")
        
        # 生成地下层系统
        map_data = generate_underground_layers(
            map_data, 
            config, 
            logger,
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.8, m) if progress_callback else None
        )
        
        # 通知地下系统生成完成
        logger.log("地下系统生成完成")
        
        # 完成进度
        if progress_callback:
            progress_callback(1.0, "地下系统生成完成")
            
    except Exception as e:
        logger.log(f"地下系统生成过程中发生错误: {str(e)}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
        if progress_callback:
            progress_callback(1.0, f"生成失败: {str(e)}")
        raise
    
    return map_data

def visualize_underground_layer(underground_layer, layer_index, map_data=None, title=None, save_path=None):
    """可视化单个地下层
    
    Args:
        underground_layer: 地下层数据
        layer_index: 层索引
        map_data: 地图数据对象，用于获取矿物层
        title: 可视化标题
        save_path: 保存路径，如果不提供则显示
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 可视化高度图
    height_map = underground_layer["height"]
    im1 = axes[0].imshow(height_map, cmap='terrain')
    axes[0].set_title(f"层{layer_index} - 地形高度")
    fig.colorbar(im1, ax=axes[0])
    
    # 可视化内容类型
    content_map = underground_layer["content"]
    im2 = axes[1].imshow(content_map, cmap='tab20')
    axes[1].set_title(f"层{layer_index} - 内容类型")
    fig.colorbar(im2, ax=axes[1])
    
    # 创建内容类型图例
    content_types = list(UndergroundContentType)
    legend_elements = []
    for i, content_type in enumerate(content_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.tab20(i / len(content_types)), 
                              markersize=10, label=content_type.name))
    
    axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 可视化矿物分布
    layer_name = f"underground_{layer_index}"
    mineral_map = None
    
    # 尝试从map_data中获取矿物层
    if map_data is not None and hasattr(map_data, 'mineral_layers') and layer_name in map_data.mineral_layers:
        mineral_map = map_data.mineral_layers[layer_name]
    # 如果不可用，尝试从underground_layer中获取
    elif "minerals" in underground_layer:
        mineral_map = underground_layer["minerals"]
    
    if mineral_map is not None:
        im3 = axes[2].imshow(mineral_map, cmap='rainbow')
        axes[2].set_title(f"层{layer_index} - 矿物分布")
        fig.colorbar(im3, ax=axes[2])
        
        # 创建矿物类型图例
        mineral_types = list(MineralType)
        legend_elements = []
        for i, mineral_type in enumerate(mineral_types):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.rainbow(i / len(mineral_types)), 
                                  markersize=10, label=mineral_type.name))
        
        axes[2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[2].text(0.5, 0.5, "矿物数据不可用", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes)
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
# 添加获取地下数据信息的辅助函数
def get_underground_statistics(map_data):
    """计算地下系统统计信息
    
    Args:
        map_data: MapData实例
        
    Returns:
        包含统计信息的字典
    """
    stats = {
        "layers": len(map_data.underground_layers),
        "structures": 0,
        "minerals": {},
        "content_types": {},
        "water_features": {
            "rivers": 0,
            "lakes": 0,
            "seepage_areas": 0
        }
    }
    
    # 统计结构
    if hasattr(map_data, "underground_structures"):
        stats["structures"] = sum(len(structs) for structs in map_data.underground_structures.values())
    
    # 统计矿物和内容类型
    for layer_name, layer in map_data.underground_layers.items():
        if "minerals" in layer:
            minerals = np.array(layer["minerals"])
            mineral_counts = {}
            for mineral_type in MineralType:
                count = np.sum(minerals == mineral_type)
                if count > 0:
                    mineral_counts[mineral_type.name] = count
            stats["minerals"][layer_name] = mineral_counts
        
        if "content" in layer:
            content = np.array(layer["content"])
            content_counts = {}
            for content_type in UndergroundContentType:
                count = np.sum(content == content_type)
                if count > 0:
                    content_counts[content_type.name] = count
            stats["content_types"][layer_name] = content_counts
    
    # 统计水系统特征
    if hasattr(map_data, "underground_water"):
        stats["water_features"]["rivers"] = len(map_data.underground_water.get("rivers", []))
        stats["water_features"]["lakes"] = len(map_data.underground_water.get("lakes", []))
        stats["water_features"]["seepage_areas"] = len(map_data.underground_water.get("seepage_areas", []))
    
    return stats














