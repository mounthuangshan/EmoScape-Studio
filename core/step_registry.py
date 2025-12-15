from core.generation_steps import GenerationStep, StepManager

def register_all_steps(manager: StepManager):
    """注册所有地图生成步骤"""
    
    # 1. 数据加载
    manager.register_step(GenerationStep(
        id="data_loading",
        name="数据加载",
        description="加载地图生成所需的基础数据和配置",
        provides=["objects_db", "biome_data"],
        execute_func=execute_data_loading,
        ui_config={
            "icon": "database",
            "color": "#6baed6",
            "group": "基础步骤"
        }
    ))
    
    # 2. 参数映射
    manager.register_step(GenerationStep(
        id="preference_mapping",
        name="参数映射",
        description="将用户偏好设置映射到地图生成参数",
        dependencies=["data_loading"],
        requires=["preferences"],
        provides=["map_params", "terrain_params"],
        execute_func=execute_preference_mapping,
        ui_config={
            "icon": "sliders",
            "color": "#9ecae1",
            "group": "基础步骤"
        }
    ))
    
    # 3. 地形生成
    manager.register_step(GenerationStep(
        id="terrain_generation",
        name="地形生成",
        description="生成基础高度图",
        dependencies=["preference_mapping"],
        requires=["terrain_params", "width", "height"],
        provides=["height_map"],
        execute_func=execute_terrain_generation,
        ui_config={
            "icon": "mountain",
            "color": "#a1d99b",
            "group": "地形步骤"
        }
    ))
    
    # 4. 地理数据导入（可选，与地形生成二选一）
    manager.register_step(GenerationStep(
        id="geo_data_import",
        name="地理数据导入",
        description="从真实地理数据导入地形",
        dependencies=["preference_mapping"],
        requires=["width", "height"],
        provides=["height_map"],
        enabled=False,  # 默认禁用
        optional=True,
        execute_func=execute_geo_data_import,
        ui_config={
            "icon": "globe",
            "color": "#bdbdbd",
            "group": "地形步骤"
        }
    ))
    
    # 5. 温度生成
    manager.register_step(GenerationStep(
        id="temperature_generation",
        name="温度生成",
        description="基于高度图生成温度分布",
        dependencies=["terrain_generation", "geo_data_import"],  # 依赖两个互斥的步骤之一
        requires=["height_map", "width", "height", "terrain_params"],
        provides=["temperature_map"],
        execute_func=execute_temperature_generation,
        ui_config={
            "icon": "thermometer",
            "color": "#fd8d3c",
            "group": "气候步骤"
        }
    ))
    
    # 6. 湿度生成
    manager.register_step(GenerationStep(
        id="humidity_generation",
        name="湿度生成",
        description="基于高度图和温度图生成湿度分布",
        dependencies=["temperature_generation"],
        requires=["height_map", "temperature_map", "width", "height", "terrain_params"],
        provides=["humidity_map"],
        execute_func=execute_humidity_generation,
        ui_config={
            "icon": "droplet",
            "color": "#74c476",
            "group": "气候步骤"
        }
    ))
    
    # 7. 高度编辑
    manager.register_step(GenerationStep(
        id="height_editing",
        name="高度编辑",
        description="手动调整地形高度",
        dependencies=["humidity_generation"],
        requires=["height_map", "temperature_map", "humidity_map"],
        provides=["edited_height_map", "edited_temp_map", "edited_humid_map"],
        optional=True,  # 可选步骤
        execute_func=execute_height_editing,
        ui_config={
            "icon": "edit",
            "color": "#969696",
            "group": "编辑步骤",
            "interactive": True  # 标记为需要用户交互的步骤
        }
    ))
    
    # 8. 洞穴和峡谷生成
    manager.register_step(GenerationStep(
        id="caves_generation",
        name="洞穴和峡谷生成",
        description="在地形中生成洞穴和峡谷",
        dependencies=["height_editing"],
        requires=["edited_height_map"],
        provides=["carved_height_map", "cave_data"],
        execute_func=execute_caves_generation,
        ui_config={
            "icon": "archive",
            "color": "#525252",
            "group": "地形特征步骤"
        }
    ))
    
    # 9. 河流系统生成
    manager.register_step(GenerationStep(
        id="river_generation",
        name="河流系统生成",
        description="生成河流网络并更新高度图",
        dependencies=["caves_generation"],
        requires=["carved_height_map", "width", "height", "terrain_params"],
        provides=["river_map", "river_features", "river_points", "updated_height_map"],
        execute_func=execute_river_generation,
        ui_config={
            "icon": "water",
            "color": "#3182bd",
            "group": "地形特征步骤"
        }
    ))
    
    # 10. 气候更新
    manager.register_step(GenerationStep(
        id="climate_update",
        name="气候更新",
        description="根据修改后的地形更新温度和湿度",
        dependencies=["river_generation"],
        requires=["updated_height_map", "width", "height", "terrain_params"],
        provides=["updated_temp_map", "updated_humid_map"],
        execute_func=execute_climate_update,
        ui_config={
            "icon": "cloud",
            "color": "#9ecae1",
            "group": "气候步骤"
        }
    ))
    
    # 11. 生态系统模拟
    manager.register_step(GenerationStep(
        id="ecosystem_simulation",
        name="生态系统模拟",
        description="应用生态系统动态模拟增强气候",
        dependencies=["climate_update"],
        requires=["updated_height_map", "updated_temp_map", "updated_humid_map"],
        provides=["enhanced_temp_map", "enhanced_humid_map"],
        execute_func=execute_ecosystem_simulation,
        ui_config={
            "icon": "leaf",
            "color": "#31a354",
            "group": "生态步骤"
        }
    ))
    
    # 12. 微气候生成
    manager.register_step(GenerationStep(
        id="microclimate_generation",
        name="微气候生成",
        description="生成局部微气候区域",
        dependencies=["ecosystem_simulation"],
        requires=["updated_height_map", "enhanced_temp_map", "enhanced_humid_map"],
        provides=["microclimate_map", "final_temp_map", "final_humid_map"],
        execute_func=execute_microclimate_generation,
        ui_config={
            "icon": "sun",
            "color": "#fdae6b",
            "group": "生态步骤"
        }
    ))
    
    # 13. 生物群落分类
    manager.register_step(GenerationStep(
        id="biome_classification",
        name="生物群落分类",
        description="分配生物群落类型",
        dependencies=["microclimate_generation"],
        requires=["updated_height_map", "final_temp_map", "final_humid_map", "biome_data"],
        provides=["biome_map"],
        execute_func=execute_biome_classification,
        ui_config={
            "icon": "grid",
            "color": "#74c476",
            "group": "生态步骤"
        }
    ))
    
    # 14. 生物群系过渡处理
    manager.register_step(GenerationStep(
        id="biome_transitions",
        name="生物群系过渡处理",
        description="创建生物群系间的自然过渡区",
        dependencies=["biome_classification"],
        requires=["biome_map", "updated_height_map", "final_temp_map", "final_humid_map", "biome_data"],
        provides=["transition_biome_map"],
        execute_func=execute_biome_transitions,
        ui_config={
            "icon": "git-branch",
            "color": "#a1d99b",
            "group": "生态步骤"
        }
    ))
    
    # 15. 植被和建筑放置
    manager.register_step(GenerationStep(
        id="vegetation_buildings",
        name="植被和建筑放置",
        description="放置植被和建筑物",
        dependencies=["biome_transitions"],
        requires=["transition_biome_map", "updated_height_map", "river_map", "preferences", "map_params"],
        provides=["vegetation", "buildings", "roads", "settlements"],
        execute_func=execute_vegetation_buildings,
        ui_config={
            "icon": "home",
            "color": "#9e9ac8",
            "group": "内容步骤"
        }
    ))
    
    # 16. NEAT进化内容布局
    manager.register_step(GenerationStep(
        id="content_layout",
        name="内容布局进化",
        description="使用NEAT算法优化内容布局",
        dependencies=["vegetation_buildings"],
        requires=["buildings"],
        provides=["order", "content_layout"],
        execute_func=execute_content_layout,
        ui_config={
            "icon": "layout",
            "color": "#bcbddc",
            "group": "内容步骤"
        }
    ))
    
    # 17. 道路网络生成
    manager.register_step(GenerationStep(
        id="road_generation",
        name="道路网络生成",
        description="构建智能道路网络",
        dependencies=["content_layout"],
        requires=["updated_height_map", "buildings", "order", "river_map", "transition_biome_map", "settlements"],
        provides=["roads_network", "roads_types"],
        execute_func=execute_road_generation,
        ui_config={
            "icon": "map",
            "color": "#756bb1",
            "group": "内容步骤"
        }
    ))
    
    # 18. LLM内容与故事生成
    manager.register_step(GenerationStep(
        id="llm_processing",
        name="LLM内容生成",
        description="使用LLM生成故事和内容",
        dependencies=["road_generation"],
        requires=["content_layout", "transition_biome_map", "preferences"],
        provides=["enhanced_content_layout", "story_events"],
        execute_func=execute_llm_processing,
        ui_config={
            "icon": "book",
            "color": "#dadaeb",
            "group": "内容步骤"
        }
    ))
    
    # 19. 交互式进化
    manager.register_step(GenerationStep(
        id="interactive_evolution",
        name="交互式进化",
        description="用户评分引导的生物群系进化",
        dependencies=["llm_processing"],
        requires=["transition_biome_map", "updated_height_map", "final_temp_map", "final_humid_map"],
        provides=["evolved_biome_map"],
        optional=True,
        execute_func=execute_interactive_evolution,
        ui_config={
            "icon": "award",
            "color": "#cbc9e2",
            "group": "优化步骤",
            "interactive": True
        }
    ))
    
    # 20. 情感分析
    manager.register_step(GenerationStep(
        id="emotion_analysis",
        name="情感分析",
        description="计算地图的情感分布",
        dependencies=["interactive_evolution"],
        requires=["updated_height_map", "evolved_biome_map", "vegetation", "buildings", 
                  "river_map", "enhanced_content_layout", "cave_data", "roads", "roads_network"],
        provides=["emotion_data", "final_content_layout"],
        execute_func=execute_emotion_analysis,
        ui_config={
            "icon": "smile",
            "color": "#fee8c8",
            "group": "优化步骤"
        }
    ))

    # 注册情感分析步骤
    manager.register_step(EmotionAnalysisStep())
    

def execute_data_loading(context, params, logger):
    """执行数据加载步骤"""
    from utils.tools import load_objects_db, load_biome_config
    
    logger.log("加载对象数据库...")
    objects_db, all_objects = load_objects_db(logger)
    
    logger.log("加载生物群系配置...")
    biome_data = load_biome_config(logger)
    
    return {
        "objects_db": objects_db,
        "all_objects": all_objects,
        "biome_data": biome_data
    }

def execute_preference_mapping(context, params, logger):
    """执行参数映射步骤"""
    from utils.tools import map_preferences_to_parameters
    
    logger.log("映射用户偏好到地图参数...")
    preferences = context.get("preferences", {})
    map_params = map_preferences_to_parameters(preferences)
    
    # 扩展地形参数字典
    terrain_params = {
        "seed": map_params.get("seed"),
        "scale_factor": map_params.get("scale_factor", 2.0),
        "mountain_sharpness": map_params.get("mountain_sharpness", 1.5),
        "erosion_iterations": map_params.get("erosion_iterations", 3),
        "river_density": map_params.get("river_density", 1.0),
        "use_tectonic": map_params.get("use_tectonic", True)
    }
    
    # 直接从preferences获取地形参数
    direct_terrain_params = [
        "seed", "scale_factor", "mountain_sharpness", "erosion_iterations",
        "river_density", "use_tectonic", "detail_level", "use_frequency_optimization",
        "erosion_type", "erosion_strength", "talus_angle", "sediment_capacity",
        "rainfall", "evaporation", "min_watershed_size", "precipitation_factor",
        "meander_factor", "octaves", "persistence", "lacunarity", "plain_ratio",
        "hill_ratio", "mountain_ratio", "plateau_ratio", "latitude_effect",
        "prevailing_wind_x", "prevailing_wind_y",
        "enable_micro_detail", "enable_extreme_detection", "optimization_level"
    ]
    
    for param in direct_terrain_params:
        if param in preferences:
            terrain_params[param] = preferences[param]
    
    return {
        "map_params": map_params,
        "terrain_params": terrain_params
    }

# 实现所有剩余的执行函数
def execute_terrain_generation(context, params, logger):
    """执行地形生成步骤"""
    from core.generation.generate_height_temp_humid import generate_height_temp_humid
    
    width = context.get("width")
    height = context.get("height")
    terrain_params = context.get("terrain_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("使用算法生成基础地形...")
    
    # 过滤参数
    from generate_map import HEIGHT_GEN_PARAMS
    height_gen_params = {k: v for k, v in terrain_params.items() if k in HEIGHT_GEN_PARAMS}
    height_map = generate_height_temp_humid(width, height, **height_gen_params)
    
    # 可视化：基础高度图
    if visualize and vis:
        vis.visualize_height_map(height_map, "基础高度图", "初始地形")

    # 验证高度图
    import numpy as np
    if not isinstance(height_map, np.ndarray) or height_map.shape != (height, width):
        raise ValueError(f"地形数据维度错误: 期望({height}, {width}), 实际{height_map.shape if isinstance(height_map, np.ndarray) else type(height_map)}")
    
    return {
        "height_map": height_map
    }

def execute_geo_data_import(context, params, logger):
    """执行地理数据导入步骤"""
    from core.geo_data_importer import GeoDataImporter
    import os
    import numpy as np
    
    width = context.get("width")
    height = context.get("height")
    geo_data_path = context.get("geo_data_path")
    geo_bounds = context.get("geo_bounds")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("使用真实地理数据生成地形...")
    # 创建地理数据导入器
    geo_importer = GeoDataImporter(logger)
    
    # 导入地理数据
    if geo_data_path and os.path.exists(geo_data_path):
        # 从文件导入
        success = geo_importer.from_file(geo_data_path)
    elif geo_bounds and len(geo_bounds) == 4:
        # 下载SRTM数据
        success = geo_importer.download_srtm(geo_bounds)
    else:
        logger.log("错误: 使用真实地理数据时必须提供有效的geo_data_path或geo_bounds", "ERROR")
        success = False
    
    if success:
        # 处理并获取高度图
        height_map = geo_importer.get_height_map(width, height, normalize_range=(0.0, 1.0))
        
        # 可视化：原始高度图
        if visualize and height_map is not None and vis:
            vis.visualize_height_map(height_map, "原始地理数据高度图", "初始地形")
        
        if height_map is not None:
            # 确保高度图尺寸正确
            if height_map.shape != (height, width):
                logger.log(f"调整高度图尺寸从 {height_map.shape} 到 ({height}, {width})", "INFO")
                from scipy.ndimage import zoom
                
                # 计算缩放因子
                zoom_factors = (height / height_map.shape[0], width / height_map.shape[1])
                
                # 使用双三次插值调整高度图
                try:
                    height_map = zoom(height_map, zoom_factors, order=3)
                    logger.log(f"高度图调整成功，当前尺寸: {height_map.shape}", "INFO")
                except Exception as zoom_error:
                    logger.log(f"高度图调整失败: {zoom_error}", "ERROR")
                    height_map = None

            # 可视化：调整后的高度图
            if visualize and height_map is not None and vis:
                vis.visualize_height_map(height_map, "调整后高度图", "地形处理")
        
        logger.log("成功从真实地理数据生成地形")
        return {"height_map": height_map}
    else:
        logger.log("地理数据导入失败", "WARNING")
        return {"height_map": None}

def execute_temperature_generation(context, params, logger):
    """执行温度生成步骤"""
    from core.generation.generate_temperature_map import generate_temperature_map
    
    height_map = context.get("height_map")
    width = context.get("width")
    height = context.get("height")
    terrain_params = context.get("terrain_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("生成温度分布...")
    temp_map = generate_temperature_map(
        height_map, 
        width, 
        height,
        seed=terrain_params.get("seed"),
        latitude_effect=terrain_params.get("latitude_effect", 0.5),
        use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
    )
    
    # 可视化：温度图
    if visualize and vis:
        vis.visualize_temperature_map(temp_map, "温度分布", "气候生成")
    
    return {
        "temperature_map": temp_map
    }

def execute_humidity_generation(context, params, logger):
    """执行湿度生成步骤"""
    from core.generation.generate_humidity_map import generate_humidity_map
    
    height_map = context.get("height_map")
    temperature_map = context.get("temperature_map")
    width = context.get("width")
    height = context.get("height")
    terrain_params = context.get("terrain_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("生成湿度分布...")
    wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
    wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
    
    humid_map = generate_humidity_map(
        height_map,
        width,
        height,
        temp_map=temperature_map,
        seed=terrain_params.get("seed"),
        prevailing_wind=(wind_x, wind_y),
        use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
    )
    
    # 可视化：湿度图
    if visualize and vis:
        vis.visualize_humidity_map(humid_map, "湿度分布", "气候生成")
    
    return {
        "humidity_map": humid_map
    }

def execute_height_editing(context, params, logger):
    """执行高度编辑步骤"""
    from utils.tools import manually_adjust_height
    
    height_map = context.get("height_map")
    temp_map = context.get("temperature_map")
    humid_map = context.get("humidity_map")
    map_params = context.get("map_params", {})
    parent_frame = context.get("parent_frame")
    use_gui_editors = context.get("use_gui_editors", False)
    callback = context.get("callback")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    # 创建临时MapData对象用于编辑
    from core.core import MapData, MapGenerationConfig
    width = context.get("width")
    height = context.get("height")
    config = context.get("config") or MapGenerationConfig(width=width, height=height)
    map_data = MapData(width, height, config)
    map_data.layers["height"] = height_map
    map_data.layers["temperature"] = temp_map
    map_data.layers["humidity"] = humid_map
    
    # 使用GUI集成编辑器的情况
    if parent_frame and use_gui_editors:
        logger.log("等待GUI编辑器完成高度调整...")
        
        # 在map_data中存储生成状态
        map_data.pending_editor = "height_editor"
        map_data.editor_state = {
            'width': width,
            'height': height,
            'terrain_params': context.get("terrain_params", {}),
            'map_params': map_params
        }
        map_data.generation_state = {
            'resume_point': 'post_height_edit',
            'width': width,
            'height': height,
            'terrain_params': context.get("terrain_params", {}),
            'map_params': map_params
        }
        
        # 确保标记为未完成
        map_data.generation_complete = False
        
        # 通知上层应用
        if callback:
            callback({
                'action': 'show_editor',
                'editor_type': 'height_editor',
                'map_data': map_data,
                'state': map_data.editor_state
            })
        
        # 返回一个特殊值，表示需要暂停执行
        return {"needs_interaction": True, "map_data": map_data}
    else:
        # 独立窗口模式
        try:
            logger.log("启动高度手动调整功能...")
            hand_map_data, hand_height_map, hand_temp_map, hand_humid_map = manually_adjust_height(
                map_data, map_params, logger, seed=map_params.get("seed")
            )
            logger.log("手动调整完成")
            
            # 可视化：手动调整后的高度图
            if visualize and vis:
                vis.visualize_height_map(hand_height_map, "手动调整后高度图", "高度修正")
                
            return {
                "edited_height_map": hand_height_map,
                "edited_temp_map": hand_temp_map,
                "edited_humid_map": hand_humid_map
            }
        except Exception as e:
            logger.log(f"手动调整功能出错: {e}", "WARNING")
            return {
                "edited_height_map": height_map,
                "edited_temp_map": temp_map,
                "edited_humid_map": humid_map
            }

def execute_caves_generation(context, params, logger):
    """执行洞穴和峡谷生成步骤"""
    from core.generation.carve_caves_and_ravines import carve_caves_and_ravines
    
    height_map = context.get("edited_height_map") or context.get("height_map")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("处理地理特征：洞穴和峡谷...")
    carved_height_map, cave_data = carve_caves_and_ravines(height_map)
    
    # 可视化：洞穴和峡谷后的高度图
    if visualize and vis:
        vis.visualize_height_map(carved_height_map, "洞穴和峡谷后高度图", "地形特征")
    
    return {
        "carved_height_map": carved_height_map,
        "cave_data": cave_data
    }

def execute_river_generation(context, params, logger):
    """执行河流系统生成步骤"""
    from core.generation.generate_rivers import generate_rivers_map
    import numpy as np
    
    height_map = context.get("carved_height_map")
    width = context.get("width")
    height = context.get("height")
    terrain_params = context.get("terrain_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("生成河流系统...")
    # 获取河流参数
    min_watershed_size = terrain_params.get("min_watershed_size", 50)
    precipitation_factor = terrain_params.get("precipitation_factor", 1.0)
    meander_factor = terrain_params.get("meander_factor", 0.3)

    # 确保有一个默认种子值
    river_seed = terrain_params.get("seed")
    if river_seed is None:
        river_seed = np.random.randint(1, 10000)
        logger.log(f"为河流生成随机种子: {river_seed}")

    # 生成河流并更新高度图
    rivers_map, river_features, updated_height_map = generate_rivers_map(
        height_map,
        width,
        height,
        seed=river_seed,
        min_watershed_size=min_watershed_size,
        precipitation_factor=precipitation_factor,
        meander_factor=meander_factor
    )
    
    # 可视化：河流系统和更新后的高度图
    if visualize and vis:
        vis.visualize_rivers(updated_height_map, rivers_map, "河流系统", "地形特征")
        vis.visualize_height_map(updated_height_map, "河流侵蚀后高度图", "地形特征")
    
    # 转换河流数据结构
    rivers_map_bool = np.zeros((height, width), dtype=bool)
    river_points = []
    
    if river_features:
        for river_path in river_features:
            for y, x in river_path:
                if 0 <= y < height and 0 <= x < width:
                    rivers_map_bool[y, x] = True
                    river_points.append((x, y))
    
    return {
        "river_map": rivers_map_bool,
        "river_features": river_features,
        "river_points": river_points,
        "updated_height_map": updated_height_map
    }

def execute_climate_update(context, params, logger):
    """执行气候更新步骤"""
    from core.generation.generate_temperature_map import generate_temperature_map
    from core.generation.generate_humidity_map import generate_humidity_map
    
    updated_height_map = context.get("updated_height_map")
    width = context.get("width")
    height = context.get("height")
    terrain_params = context.get("terrain_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("更新温度分布...")
    updated_temp_map = generate_temperature_map(
        updated_height_map, 
        width, 
        height,
        seed=terrain_params.get("seed"),
        latitude_effect=terrain_params.get("latitude_effect", 0.5),
        use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
    )
    
    # 可视化：温度图
    if visualize and vis:
        vis.visualize_temperature_map(updated_temp_map, "更新后温度分布", "气候生成")
    
    logger.log("更新湿度分布...")
    wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
    wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
    
    updated_humid_map = generate_humidity_map(
        updated_height_map,
        width,
        height,
        temp_map=updated_temp_map,
        seed=terrain_params.get("seed"),
        prevailing_wind=(wind_x, wind_y),
        use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
    )
    
    # 可视化：湿度图
    if visualize and vis:
        vis.visualize_humidity_map(updated_humid_map, "更新后湿度分布", "气候生成")
    
    return {
        "updated_temp_map": updated_temp_map,
        "updated_humid_map": updated_humid_map
    }

def execute_ecosystem_simulation(context, params, logger):
    """执行生态系统模拟步骤"""
    from core.generation.ecosystem_dynamics import simulate_ecosystem_dynamics
    
    updated_height_map = context.get("updated_height_map")
    updated_temp_map = context.get("updated_temp_map")
    updated_humid_map = context.get("updated_humid_map")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("应用生态系统动态模拟...")
    enhanced_temp_map, enhanced_humid_map = simulate_ecosystem_dynamics(
        updated_height_map, updated_temp_map, updated_humid_map, iterations=3
    )
    
    # 可视化：生态系统模拟后的气候
    if visualize and vis:
        vis.visualize_temperature_map(enhanced_temp_map, "生态系统模拟后温度", "生态系统")
        vis.visualize_humidity_map(enhanced_humid_map, "生态系统模拟后湿度", "生态系统")
    
    return {
        "enhanced_temp_map": enhanced_temp_map,
        "enhanced_humid_map": enhanced_humid_map
    }

def execute_microclimate_generation(context, params, logger):
    """执行微气候生成步骤"""
    from core.generation.microclimates import generate_microclimates
    from core.services.analyze_map_emotions import analyze_terrain_quality
    
    updated_height_map = context.get("updated_height_map")
    enhanced_temp_map = context.get("enhanced_temp_map")
    enhanced_humid_map = context.get("enhanced_humid_map")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("生成微气候区域...")
    microclimate_map, final_temp_map, final_humid_map = generate_microclimates(
        updated_height_map, enhanced_temp_map, enhanced_humid_map
    )
    
    # 可视化：微气候
    if visualize and vis:
        vis.visualize_microclimate(microclimate_map, updated_height_map, "微气候区域", "生态系统")
    
    # 分析地形质量
    quality_metrics, suggestions = analyze_terrain_quality(updated_height_map, final_temp_map, final_humid_map)
    for suggestion in suggestions:
        logger.log(f"地形建议: {suggestion}", "INFO")
    
    return {
        "microclimate_map": microclimate_map,
        "final_temp_map": final_temp_map,
        "final_humid_map": final_humid_map,
        "quality_metrics": quality_metrics,
        "quality_suggestions": suggestions
    }

def execute_biome_classification(context, params, logger):
    """执行生物群落分类步骤"""
    from utils.classify_biome import classify_biome
    import numpy as np
    
    updated_height_map = context.get("updated_height_map")
    final_temp_map = context.get("final_temp_map")
    final_humid_map = context.get("final_humid_map")
    biome_data = context.get("biome_data")
    width = context.get("width")
    height = context.get("height")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("分配生物群落...")
    biome_map = classify_biome(updated_height_map, final_temp_map, final_humid_map, biome_data)
    
    # 可视化：生物群系分类
    if visualize and vis:
        vis.visualize_biome_map(biome_map, biome_data, "生物群系分类", "生态分布")
    
    # 验证生物群系数据
    if not isinstance(biome_map, np.ndarray):
        raise ValueError("生物群系数据必须为NumPy数组")
    if biome_map.shape != (height, width):
        raise ValueError(f"生物群系数据维度错误: 预期({height}, {width}), 实际{biome_map.shape}")
    
    return {
        "biome_map": biome_map
    }

def execute_biome_transitions(context, params, logger):
    """执行生物群系过渡区处理步骤"""
    from utils.biome_transitions import create_biome_transitions
    
    biome_map = context.get("biome_map")
    updated_height_map = context.get("updated_height_map")
    final_temp_map = context.get("final_temp_map")
    final_humid_map = context.get("final_humid_map")
    biome_data = context.get("biome_data")
    
    logger.log("创建生物群系过渡区...")
    transition_biome_map = create_biome_transitions(
        biome_map, updated_height_map, final_temp_map, final_humid_map, biome_data
    )
    
    return {
        "transition_biome_map": transition_biome_map
    }

def execute_vegetation_buildings(context, params, logger):
    """执行植被和建筑放置步骤"""
    from utils.tools import place_vegetation_and_buildings
    
    transition_biome_map = context.get("transition_biome_map")
    updated_height_map = context.get("updated_height_map")
    river_map = context.get("river_map")
    preferences = context.get("preferences", {})
    map_params = context.get("map_params", {})
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("放置植被与建筑...")
    try:
        vegetation, buildings, roads, settlements = place_vegetation_and_buildings(
            transition_biome_map, 
            updated_height_map, 
            river_map, 
            preferences, 
            map_params
        )
    except Exception as e:
        logger.log(f"植被和建筑生成错误: {e}", "ERROR")
        # 提供默认值避免后续过程崩溃
        vegetation, buildings, roads, settlements = [], [], [], []
    
    # 可视化：植被和建筑
    if visualize and vis:
        vis.visualize_objects(updated_height_map, vegetation, buildings, roads, "植被与建筑", "对象放置")
    
    return {
        "vegetation": vegetation,
        "buildings": buildings,
        "roads": roads,
        "settlements": settlements
    }

def execute_content_layout(context, params, logger):
    """执行内容布局进化步骤"""
    from core.services.run_neat_for_content_layout import run_neat_for_content_layout
    
    buildings = context.get("buildings", [])
    
    logger.log("NEAT进化内容布局...")
    try:
        order, content_layout = run_neat_for_content_layout(buildings)
    except Exception as e:
        logger.log(f"内容布局生成错误: {e}", "ERROR")
        order, content_layout = [], {}
    
    return {
        "order": order,
        "content_layout": content_layout
    }

def execute_road_generation(context, params, logger):
    """执行道路网络生成步骤"""
    from core.generation.build_hierarchical_road_network import build_roads
    import numpy as np
    
    updated_height_map = context.get("updated_height_map")
    buildings = context.get("buildings", [])
    order = context.get("order", [])
    river_map = context.get("river_map")
    transition_biome_map = context.get("transition_biome_map")
    settlements = context.get("settlements", [])
    width = context.get("width")
    height = context.get("height")
    visualize = context.get("visualize", False)
    vis = context.get("vis")
    
    logger.log("构建智能道路网络...")
    try:
        roads_network, roads_types = build_roads(
            updated_height_map, 
            buildings, 
            order,
            river_map,
            transition_biome_map,
            settlements
        )
    except Exception as e:
        logger.log(f"道路网络生成错误: {e}", "ERROR")
        # 提供默认数组避免后续过程崩溃
        roads_network = np.zeros((height, width), dtype=np.uint8)
        roads_types = np.zeros((height, width), dtype=np.uint8)
    
    # 可视化：道路网络
    if visualize and vis:
        road_points = []
        if roads_network is not None:
            for y in range(height):
                for x in range(width):
                    if roads_network[y, x] > 0:
                        road_points.append({"x": x, "y": y})
        # 添加检查确保有足够数据显示
        if len(road_points) > 0:
            vis.visualize_objects(updated_height_map, None, None, road_points, "道路网络", "交通系统")
        else:
            logger.log("警告: 道路网络为空，跳过可视化", "WARNING")
    
    return {
        "roads_network": roads_network,
        "roads_types": roads_types
    }

def execute_llm_processing(context, params, logger):
    """执行LLM内容生成步骤"""
    from utils.llm import LLMIntegration
    
    content_layout = context.get("content_layout", {})
    transition_biome_map = context.get("transition_biome_map")
    preferences = context.get("preferences", {})
    
    logger.log("LLM处理内容与故事生成...")
    try:
        llm = LLMIntegration(logger)
        
        if not content_layout:
            content_layout = {"objects": []}
        elif "objects" not in content_layout:
            content_layout["objects"] = []

        # 调用LLM处理任务
        enhanced_content_layout = llm.process_llm_tasks(
            content_layout,
            transition_biome_map,
            preferences
        )
        
        # 提取故事事件
        story_events = []
        if "story_events" in enhanced_content_layout:
            for event in enhanced_content_layout["story_events"]:
                if isinstance(event, dict) and "x" in event and "y" in event:
                    try:
                        event["x"] = int(event["x"])
                        event["y"] = int(event["y"])
                        story_events.append(event)
                    except (ValueError, TypeError):
                        pass

    except Exception as e:
        logger.log(f"LLM内容生成错误: {e}", "ERROR")
        enhanced_content_layout = content_layout
        story_events = []

    return {
        "enhanced_content_layout": enhanced_content_layout,
        "story_events": story_events
    }

def execute_interactive_evolution(context, params, logger):
    """执行交互式进化步骤"""
    from core.evolution.evolve_generation import BiomeEvolutionEngine
    import numpy as np
    
    transition_biome_map = context.get("transition_biome_map")
    updated_height_map = context.get("updated_height_map")
    final_temp_map = context.get("final_temp_map")
    final_humid_map = context.get("final_humid_map")
    map_params = context.get("map_params", {})
    parent_frame = context.get("parent_frame")
    use_gui_editors = context.get("use_gui_editors", False)
    callback = context.get("callback")
    visualize = context.get("visualize", False)
    
    logger.log("准备交互式进化评分...")
    try:
        # 初始化进化引擎
        engine = BiomeEvolutionEngine(
            biome_map=transition_biome_map,
            height_map=updated_height_map,
            temperature_map=final_temp_map,
            moisture_map=final_humid_map,
            memory_path="./data/models/evolution_memory.dat"
        )
        
        # 从map_params获取进化代数，不存在则默认为1
        generations = map_params.get("evolution_generations", 1)
        logger.log(f"将进行 {generations} 代进化...")
        
        # 检查是否支持交互式模式
        is_interactive = map_params.get("interactive_evolution", True)
        
        if is_interactive and parent_frame and use_gui_editors:
            # GUI集成模式 - 保存状态并暂停执行
            logger.log("等待GUI界面完成评分...")
            
            # 创建临时MapData对象用于编辑
            from core.core import MapData, MapGenerationConfig
            width = context.get("width")
            height = context.get("height")
            config = context.get("config") or MapGenerationConfig(width=width, height=height)
            map_data = MapData(width, height, config)
            
            # 保存到map_data中供GUI使用
            map_data.pending_editor = "evolution_scorer"
            map_data.editor_state = {
                'engine': engine,
                'generations': generations,
                'map_params': map_params
            }
            
            # 通知上层应用
            if callback:
                callback({
                    'action': 'show_editor',
                    'editor_type': 'evolution_scorer',
                    'map_data': map_data,
                    'state': map_data.editor_state
                })
            
            # 返回一个特殊值，表示需要暂停执行
            return {"needs_interaction": True, "map_data": map_data}
        else:
            # 非交互模式或独立窗口模式
            logger.log("使用自动评分模式...")
            user_scores = [np.random.uniform(5, 8) for _ in range(engine.population_size)]
            
            # 执行进化
            for gen in range(generations):
                logger.log(f"执行第 {gen+1}/{generations} 代进化")
                engine.evolve_generation(user_scores)
                
                # 如果不是最后一代，生成新的自动评分
                if gen < generations - 1:
                    user_scores = engine.auto_score_population()
            
            # 获取最终的生物群系图
            evolved_biome_map = engine.best_individual
            
            if evolved_biome_map is not None:
                logger.log("进化产生了有效的生物群系图")
                return {"evolved_biome_map": evolved_biome_map}
            else:
                logger.log("进化未产生有效结果，保持原始生物群系", "WARNING")
                return {"evolved_biome_map": transition_biome_map}
    except Exception as e:
        logger.log(f"交互式进化错误: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "DEBUG")
        return {"evolved_biome_map": transition_biome_map}

def execute_emotion_analysis(context, params, logger):
    """执行情感分析步骤"""
    from core.services.analyze_map_emotions import MapEmotionAnalyzer
    
    updated_height_map = context.get("updated_height_map") 
    evolved_biome_map = context.get("evolved_biome_map") or context.get("transition_biome_map")
    vegetation = context.get("vegetation", [])
    buildings = context.get("buildings", [])
    river_map = context.get("river_map")
    enhanced_content_layout = context.get("enhanced_content_layout", {})
    cave_data = context.get("cave_data", {})
    roads = context.get("roads", [])
    roads_network = context.get("roads_network")
    
    # 提取洞穴入口
    cave_entrances = []
    if isinstance(cave_data, dict) and "entrances" in cave_data:
        cave_entrances = cave_data["entrances"]
    
    logger.log("计算情感信息...")
    try:
        # 创建分析器实例
        analyzer = MapEmotionAnalyzer(
            updated_height_map, 
            evolved_biome_map, 
            vegetation, 
            buildings, 
            river_map, 
            enhanced_content_layout, 
            cave_entrances,
            roads,
            roads_network
        )
        
        # 分析情感信息
        final_content_layout = analyzer.analyze_map_emotions(
            evolved_biome_map, vegetation, buildings, river_map, 
            enhanced_content_layout, cave_entrances, roads, roads_network
        )
        
        # 提取情感数据
        emotion_data = {
            "emotion_map": analyzer.get_emotion_heatmap() if hasattr(analyzer, "get_emotion_heatmap") else None,
            "primary_emotions": analyzer.get_primary_emotions() if hasattr(analyzer, "get_primary_emotions") else {},
            "emotion_summary": analyzer.get_summary() if hasattr(analyzer, "get_summary") else ""
        }
        
        logger.log(f"情感分析完成: {emotion_data.get('emotion_summary', '无摘要')}")
        
    except Exception as e:
        logger.log(f"情感分析错误: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "DEBUG")
        final_content_layout = enhanced_content_layout
        emotion_data = {}
    
    return {
        "emotion_data": emotion_data,
        "final_content_layout": final_content_layout
    }
    
    
# 情感分析步骤
class EmotionAnalysisStep(GenerationStep):
    def __init__(self):
        super().__init__(
            id="emotion_analysis",
            name="情感分析",
            description="分析地图各区域的情感特征",
            dependencies=["biome_classification", "vegetation_placement", "road_generation"]
        )
    
    def execute(self, context, logger):
        logger.log("执行情感分析...\n")
        
        try:
            # 获取必要数据
            map_data = context.get("map_data")
            if not map_data or not map_data.is_valid():
                raise ValueError("无效的地图数据，无法执行情感分析")
            
            # 使用情感管理器
            from utils.emotion_manager import EmotionManager
            emotion_manager = EmotionManager(logger)
            success = emotion_manager.analyze_map_emotions(map_data)
            
            if success:
                logger.log("情感分析完成")
                # 情感管理器已经将结果保存到map_data.content_layout
                context["emotion_data"] = map_data.content_layout.get("emotions", {})
            else:
                logger.log("情感分析失败", "WARNING")
            
        except Exception as e:
            logger.log(f"情感分析步骤出错: {e}", "ERROR")
            import traceback
            logger.log(traceback.format_exc(), "DEBUG")
        
        return context