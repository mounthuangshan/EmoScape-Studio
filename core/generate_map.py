#from __future__ import annotations
#标准库
import hashlib
import time

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from matplotlib.gridspec import GridSpec

#网络与并发
import concurrent.futures

#其他工具

#项目文件
from core.evolution.evolve_generation import *
from core.core import *
from pygamemap import *
from core.generation.build_hierarchical_road_network import *
from core.generation.calculate_distance_to_water import *
from core.generation.carve_caves_and_ravines import *
from core.generation.generate_height_temp_humid import *
from core.generation.generate_rivers import *
from core.services.analyze_map_emotions import *
from core.services.run_neat_for_content_layout import *
from core.evolution.evolve_generation import *
from utils.classify_biome import *
from utils.llm import *
from utils.export import *
from core.geo_data_importer import GeoDataImporter
from core.services.map_tools import *
from core.generation.generate_humidity_map import *
from core.generation.generate_temperature_map import *

#########################################################################
#地图初始化
#########################################################################
#在文件顶部定义支持的参数
HEIGHT_GEN_PARAMS = [
    "seed", "scale_factor", "mountain_sharpness", "erosion_iterations",
    "use_tectonic", "detail_level", "use_frequency_optimization",
    "erosion_type", "erosion_strength", "talus_angle", "sediment_capacity",
    "octaves", "persistence", "lacunarity", "plain_ratio",
    "hill_ratio", "mountain_ratio", "plateau_ratio", 
    "enable_micro_detail", "enable_extreme_detection", "optimization_level",
    "enable_realistic_landforms",
    "large_map_mode", "province_count", "macro_feature_scale"
]

# 并行地图生成引擎
def generate_map(preferences, width, height, export_model=False, logger=None, 
                 use_real_geo_data=False, geo_data_path=None, geo_bounds=None,
                 visualize=False, visualization_path=None, 
                 parent_frame=None, parent_frame_edit=None, use_gui_editors=False,
                 map_data=None):
    """增强版地图生成函数，使用并行处理和进度跟踪，接受预先创建的map_data对象
    
    Args:
        preferences: 用户偏好设置
        width: 地图宽度
        height: 地图高度
        export_model: 是否导出3D模型
        logger: 日志记录器
        use_real_geo_data: 是否使用真实地理数据
        geo_data_path: 地理数据文件路径,如果use_real_geo_data为True且需要从文件导入
        geo_bounds: 地理边界(左,下,右,上),如果use_real_geo_data为True且需要下载SRTM数据
        visualize: 是否启用可视化
        visualization_path: 可视化图像保存路径，如果不提供则不保存
        parent_frame: 用于嵌入编辑器的Tkinter框架
        use_gui_editors: 是否使用GUI集成的编辑器
        map_data: 地图数据对象，必须提供
    """
    if map_data is None:
        raise ValueError("必须提供map_data对象")
        
    if logger is None:
        # 如果未提供 logger，则使用默认配置（无 GUI 输出）
        print("警告未提供logger")
        logger = ThreadSafeLogger(max_lines=500, log_file="map_generation.log")    
    
    # 初始化性能监控和配置
    perf = PerformanceMonitor()
    config = MapGenerationConfig(width=width, height=height, export_model=export_model)

    # 初始化可视化工具(如果启用)
    vis = None
    if visualize:
        from core.services.map_tools import MapGenerationVisualizer
        # 使用parent_frame来初始化可视化工具，实现GUI嵌入
        vis = MapGenerationVisualizer(save_path=visualization_path, show_plots=True, logger=logger, gui_frame=parent_frame)
        logger.log("已启用地图生成可视化功能")

    try:
        # 初始化llm
        llm = LLMIntegration(logger)
        # 初始化变量
        road_points = []  # 添加这一行来初始化road_points
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, config.threads)) as executor:
            try:
                max_workers = min(4, config.threads)
                assert max_workers > 0, "线程池 worker 数量必须大于0"
                
                # 检查map_data是否与请求尺寸匹配
                logger.log(f"使用已有地图数据对象 ({map_data.width}x{map_data.height})")
                if map_data.width != width or map_data.height != height:
                    logger.log(f"警告: 提供的map_data尺寸与请求尺寸不匹配", "WARNING")
                
                futures = {}
                
                # 处理恢复状态 - 从map_data中读取恢复点
                goto_post_height_edit = False
                goto_post_evolution = False
                
                if hasattr(map_data, 'generation_state') and map_data.generation_state:
                    logger.log("从之前的状态恢复地图生成...")
                    resume_point = map_data.generation_state.get('resume_point', '')
                    logger.log(f"恢复执行点: {resume_point}")
                    
                    if resume_point == 'post_height_edit':
                        goto_post_height_edit = True
                    elif resume_point == 'post_evolution':
                        goto_post_evolution = True
                
                # 启动数据加载任务
                logger.log("启动数据加载...\n")
                task_id = perf.start("data_loading")
                futures["data_loading"] = executor.submit(load_objects_db, logger)

                # 等待数据加载完成
                data, all_objs = futures["data_loading"].result()
                biome_data = load_biome_config(logger)

                perf.end(task_id)
                
                # 映射玩家偏好到参数
                task_id = perf.start("preference_mapping")
                map_params = map_preferences_to_parameters(preferences)
                
                # 扩展参数字典，添加高级参数支持
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
                    "enable_micro_detail", "enable_extreme_detection", "optimization_level",
                    "enable_realistic_landforms",
                    "large_map_mode", "province_count", "macro_feature_scale" 
                ]
                
                for param in direct_terrain_params:
                    if param in preferences:
                        terrain_params[param] = preferences[param]

                perf.end(task_id)
                
                # 检查是否有恢复的高度图数据
                height_map = None
                temp_map = None
                humid_map = None
                
                # 从map_data获取已有数据
                if "height" in map_data.layers and goto_post_height_edit:
                    height_map = map_data.get_layer("height")
                    temp_map = map_data.get_layer("temperature")
                    humid_map = map_data.get_layer("humidity")
                    logger.log("从map_data恢复高度图数据")
                
                # 如果没有从map_data恢复高度图数据，则根据情况生成
                if height_map is None and not goto_post_height_edit:
                    # 根据用户选择，使用真实地理数据或算法生成地形
                    if use_real_geo_data:
                        logger.log("使用真实地理数据生成地形...\n")
                        task_id = perf.start("geo_data_import")
                        
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
                            if visualize and height_map is not None:
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
                                if visualize and height_map is not None:
                                    vis.visualize_height_map(height_map, "调整后高度图", "地形处理")

                                # 基于高度图生成温度图
                                logger.log("生成温度分布...\n")
                                task_id = perf.start("temperature_generation")
                                temp_map = generate_temperature_map(
                                    height_map, 
                                    width, 
                                    height,
                                    seed=terrain_params.get("seed"),
                                    latitude_effect=terrain_params.get("latitude_effect", 0.5),
                                    use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                                )
                                perf.end(task_id)
                                
                                # 可视化：温度图
                                if visualize:
                                    vis.visualize_temperature_map(temp_map, "温度分布", "气候生成")
                                
                                # 基于高度图和温度图生成湿度图
                                logger.log("生成湿度分布...\n")
                                task_id = perf.start("humidity_generation")
                                wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
                                wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
                                
                                humid_map = generate_humidity_map(
                                    height_map,
                                    width,
                                    height,
                                    temp_map=temp_map,
                                    seed=terrain_params.get("seed"),
                                    prevailing_wind=(wind_x, wind_y),
                                    use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                                )
                                
                                # 可视化：湿度图
                                if visualize:
                                    vis.visualize_humidity_map(humid_map, "湿度分布", "气候生成")
                                
                                perf.end(task_id)
                                
                                map_data.layers["height"] = height_map
                                map_data.layers["temperature"] = temp_map
                                map_data.layers["humidity"] = humid_map
                            
                            logger.log("成功从真实地理数据生成地形")
                        else:
                            logger.log("地理数据导入失败，回退到算法生成", "WARNING")
                            height_map, temp_map, humid_map, rivers_map, river_features, river_points = None, None, None, None, None, []
                        
                        perf.end(task_id)
                    
                    # 如果没有使用真实地形数据或导入失败，使用算法生成
                    if height_map is None:
                        # 生成基础高度图
                        logger.log("生成基础地形...\n")
                        task_id = perf.start("terrain_generation")
                        
                        # 然后在函数调用前过滤参数
                        height_gen_params = {k: v for k, v in terrain_params.items() if k in HEIGHT_GEN_PARAMS}
                        height_map = generate_height_temp_humid(width, height, **height_gen_params)
                        
                        # 可视化：基础高度图
                        if visualize:
                            vis.visualize_height_map(height_map, "基础高度图", "初始地形")

                        # 验证高度图
                        if not isinstance(height_map, np.ndarray) or height_map.shape != (height, width):
                            raise ValueError(f"地形数据维度错误: 期望({height}, {width}), 实际{height_map.shape if isinstance(height_map, np.ndarray) else type(height_map)}")
                        
                        perf.end(task_id)

                        # 基于高度图生成温度图
                        logger.log("生成温度分布...\n")
                        task_id = perf.start("temperature_generation")
                        temp_map = generate_temperature_map(
                            height_map, 
                            width, 
                            height,
                            seed=terrain_params.get("seed"),
                            latitude_effect=terrain_params.get("latitude_effect", 0.5),
                            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                        )
                        perf.end(task_id)
                        
                        # 可视化：温度图
                        if visualize:
                            vis.visualize_temperature_map(temp_map, "温度分布", "气候生成")
                        
                        # 基于高度图和温度图生成湿度图
                        logger.log("生成湿度分布...\n")
                        task_id = perf.start("humidity_generation")
                        wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
                        wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
                        
                        humid_map = generate_humidity_map(
                            height_map,
                            width,
                            height,
                            temp_map=temp_map,
                            seed=terrain_params.get("seed"),
                            prevailing_wind=(wind_x, wind_y),
                            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                        )
                        
                        # 可视化：湿度图
                        if visualize:
                            vis.visualize_humidity_map(humid_map, "湿度分布", "气候生成")
                        
                        perf.end(task_id)
                        
                        map_data.layers["height"] = height_map
                        map_data.layers["temperature"] = temp_map
                        map_data.layers["humidity"] = humid_map
                
                # 手动调整高度 - 注意使用统一的编辑器触发逻辑
                if not goto_post_height_edit:  # 如果不是从这里恢复
                    # 更新初始基本层数据
                    map_data.layers["height"] = height_map
                    map_data.layers["temperature"] = temp_map
                    map_data.layers["humidity"] = humid_map
                    
                    # 使用GUI集成编辑器的情况
                    if parent_frame_edit and use_gui_editors:
                        logger.log("等待GUI编辑器完成高度调整...")
                        
                        # 在map_data中存储生成状态
                        map_data.pending_editor = "height_editor"
                        map_data.editor_state = {
                            'width': width,
                            'height': height,
                            'terrain_params': terrain_params,
                            'map_params': map_params
                        }
                        map_data.generation_state = {
                            'resume_point': 'post_height_edit',
                            'width': width,
                            'height': height,
                            'terrain_params': terrain_params,
                            'map_params': map_params
                        }
                        
                        # 确保标记为未完成
                        map_data.generation_complete = False
                        
                        # 返回当前地图数据对象，让上层负责显示编辑器
                        return map_data
                    else:
                        # 独立窗口模式
                        try:
                            logger.log("启动高度手动调整功能...\n")
                            hand_map_data, hand_height_map, hand_temp_map, hand_humid_map = manually_adjust_height(
                                map_data, map_params, logger, seed=map_params.get("seed")
                            )
                            logger.log("手动调整完成\n")
                            map_data = hand_map_data
                            height_map = hand_height_map
                            temp_map = hand_temp_map
                            humid_map = hand_humid_map
                        except Exception as e:
                            logger.log(f"手动调整功能出错: {e}", "WARNING")
                
                # 确保有正确的基础层数据继续后续处理
                height_map = map_data.get_layer("height")
                temp_map = map_data.get_layer("temperature")
                humid_map = map_data.get_layer("humidity")
                
                # 可视化：手动调整后的高度图
                if visualize:
                    vis.visualize_height_map(height_map, "手动调整后高度图", "高度修正")
                
                # 生成洞穴和峡谷，并进一步更新高度图
                logger.log("处理地理特征：洞穴和峡谷...\n")
                task_id = perf.start("caves_generation")
                
                carved_height_map, cave_data = carve_caves_and_ravines(height_map)
                
                # 可视化：洞穴和峡谷后的高度图
                if visualize:
                    vis.visualize_height_map(carved_height_map, "洞穴和峡谷后高度图", "地形特征")
                
                # 确保洞穴数据结构正确
                if not isinstance(cave_data, dict) or "entrances" not in cave_data:
                    logger.log("警告: 洞穴数据格式无效，使用空列表", "WARNING")
                    cave_entrances = []
                else:
                    cave_entrances = cave_data["entrances"]
                
                # 处理洞穴入口点
                formatted_caves = []
                if isinstance(cave_entrances, list):
                    for cave in cave_entrances:
                        if isinstance(cave, (list, tuple)) and len(cave) >= 2:
                            formatted_caves.append({"x": int(cave[0]), "y": int(cave[1])})
                        elif hasattr(cave, 'x') and hasattr(cave, 'y'):
                            formatted_caves.append(cave)
                
                # 更新高度图为包含洞穴的版本
                height_map = carved_height_map
                
                perf.end(task_id)
                
                # 生成河流系统并更新高度图
                logger.log("生成河流系统...\n")
                task_id = perf.start("river_generation")

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
                rivers_map, river_features, height_map = generate_rivers_map(
                    height_map,
                    width,
                    height,
                    seed=river_seed,
                    min_watershed_size=min_watershed_size,
                    precipitation_factor=precipitation_factor,
                    meander_factor=meander_factor
                )
                
                # 可视化：河流系统和更新后的高度图
                if visualize:
                    vis.visualize_rivers(height_map, rivers_map, "河流系统", "地形特征")
                    vis.visualize_height_map(height_map, "河流侵蚀后高度图", "地形特征")
                
                # 转换河流数据结构
                rivers_map = np.zeros((height, width), dtype=bool)
                river_points = []
                
                if river_features:
                    for river_path in river_features:
                        for y, x in river_path:
                            if 0 <= y < height and 0 <= x < width:
                                rivers_map[y, x] = True
                                river_points.append((x, y))
                
                perf.end(task_id)
                
                # 重新生成温度和湿度图，因为地形已经改变
                logger.log("更新温度分布...\n")
                task_id = perf.start("temperature_update")
                temp_map = generate_temperature_map(
                    height_map, 
                    width, 
                    height,
                    seed=terrain_params.get("seed"),
                    latitude_effect=terrain_params.get("latitude_effect", 0.5),
                    use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                )
                perf.end(task_id)
                
                # 可视化：温度图
                if visualize:
                    vis.visualize_temperature_map(temp_map, "更新后温度分布", "气候生成")
                
                logger.log("更新湿度分布...\n")
                task_id = perf.start("humidity_update")
                wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
                wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
                
                humid_map = generate_humidity_map(
                    height_map,
                    width,
                    height,
                    temp_map=temp_map,
                    seed=terrain_params.get("seed"),
                    prevailing_wind=(wind_x, wind_y),
                    use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
                )
                
                # 可视化：湿度图
                if visualize:
                    vis.visualize_humidity_map(humid_map, "更新后湿度分布", "气候生成")
                
                perf.end(task_id)

                # 更新地图数据对象
                map_data.layers["height"] = height_map
                map_data.layers["temperature"] = temp_map
                map_data.layers["humidity"] = humid_map
                map_data.layers["rivers"] = rivers_map
                map_data.layers["river_features"] = river_features
                map_data.layers["rivers_points"] = river_points
                map_data.layers["caves"] = formatted_caves
                
                # 生态系统模拟
                logger.log("应用生态系统动态模拟...\n")
                from core.generation.ecosystem_dynamics import simulate_ecosystem_dynamics
                enhanced_temp_map, enhanced_humid_map = simulate_ecosystem_dynamics(
                    height_map, temp_map, humid_map, iterations=3
                )

                # 将增强的气候图设置为当前使用值
                temp_map = enhanced_temp_map
                humid_map = enhanced_humid_map

                # 更新map_data中的图层
                map_data.layers["temperature"] = temp_map
                map_data.layers["humidity"] = humid_map

                # 可视化：生态系统模拟后的气候
                if visualize:
                    vis.visualize_temperature_map(temp_map, "生态系统模拟后温度", "生态系统")
                    vis.visualize_humidity_map(humid_map, "生态系统模拟后湿度", "生态系统")

                # 添加微气候系统
                logger.log("生成微气候区域...\n")
                task_id = perf.start("microclimate_generation")
                from core.generation.microclimates import generate_microclimates
                microclimate_map, micro_temp_map, micro_humid_map = generate_microclimates(
                    height_map, temp_map, humid_map
                )

                # 应用微气候效果
                temp_map = micro_temp_map
                humid_map = micro_humid_map

                # 可视化：微气候
                if visualize:
                    vis.visualize_microclimate(microclimate_map, height_map, "微气候区域", "生态系统")

                # 存储微气候图层
                map_data.layers["microclimate"] = microclimate_map
                map_data.layers["temperature"] = temp_map
                map_data.layers["humidity"] = humid_map

                perf.end(task_id)

                # 分析地形质量
                quality_metrics, suggestions = analyze_terrain_quality(height_map, temp_map, humid_map)
                for suggestion in suggestions:
                    logger.log(f"地形建议: {suggestion}", "INFO")
                
                # 分类生物群落
                logger.log("分配生物群落...\n")
                task_id = perf.start("biome_classification")
                biome_map = classify_biome(height_map, temp_map, humid_map, biome_data)

                # 可视化：生物群系分类
                if visualize:
                    vis.visualize_biome_map(biome_map, biome_data, "生物群系分类", "生态分布")

                # 验证生物群系数据
                if not isinstance(biome_map, np.ndarray):
                    raise ValueError("生物群系数据必须为NumPy数组")
                if biome_map.shape != (height, width):
                    raise ValueError(f"生物群系数据维度错误: 预期({height}, {width}), 实际{biome_map.shape}")

                # 统一创建并填充生物群系图层
                if "biome" not in map_data.layers:
                    map_data.create_layer("biome", dtype=np.int32, fill_value=0)

                # 修改此处以处理CuPy/NumPy兼容性问题
                try:
                    # 检查是否是GPU数组
                    if hasattr(map_data.layers["biome"], 'get') and hasattr(map_data.layers["biome"], 'device'):
                        # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                        logger.log("将GPU数组转换成CPU数组")
                        import cupy as cp
                        cp_biome_map = cp.asarray(biome_map, dtype=cp.int32)
                        map_data.layers["biome"][:] = cp_biome_map
                    else:
                        # 如果是CPU数组(NumPy)，直接赋值
                        logger.log("是CPU数组，直接赋值")
                        map_data.layers["biome"][:] = biome_map.astype(np.int32)
                except Exception as e:
                    logger.log(f"生物群系数据赋值失败: {e}", "ERROR")
                    # 降级方案：逐个元素复制
                    try:
                        h, w = biome_map.shape
                        for y in range(h):
                            for x in range(w):
                                map_data.layers["biome"][y, x] = int(biome_map[y, x])
                        logger.log("使用逐元素复制方法完成生物群系数据赋值", "WARNING")
                    except Exception as e2:
                        logger.log(f"退化方案也失败: {e2}", "ERROR")
                        raise

                perf.end(task_id)
                
                # 在生物群系分类后添加过渡区处理
                logger.log("创建生物群系过渡区...\n")
                from utils.biome_transitions import create_biome_transitions
                transition_biome_map = create_biome_transitions(
                    biome_map, height_map, temp_map, humid_map, biome_data
                )
                        
                # 使用带有过渡区的生物群系图替换原来的图
                try:
                    # 检查是否是GPU数组
                    if hasattr(map_data.layers["biome"], 'get') and hasattr(map_data.layers["biome"], 'device'):
                        # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                        logger.log("将GPU数组转换成CPU数组")
                        import cupy as cp
                        cp_transition_map = cp.asarray(transition_biome_map, dtype=cp.int32)
                        map_data.layers["biome"][:] = cp_transition_map
                    else:
                        # 如果是CPU数组(NumPy)，直接赋值
                        logger.log("是CPU数组，直接赋值")
                        map_data.layers["biome"][:] = transition_biome_map.astype(np.int32)
                except Exception as e:
                    logger.log(f"生物群系过渡区数据赋值失败: {e}", "ERROR")
                    # 降级方案：逐个元素复制
                    try:
                        h, w = transition_biome_map.shape
                        for y in range(h):
                            for x in range(w):
                                map_data.layers["biome"][y, x] = int(transition_biome_map[y, x])
                        logger.log("使用逐元素复制方法完成生物群系过渡区赋值", "WARNING")
                    except Exception as e2:
                        logger.log(f"退化方案也失败: {e2}", "ERROR")
                        raise

                biome_map = transition_biome_map
                
                #转换变量名，后面用的都是rivers
                rivers = rivers_map
                                   
                # 放置植被和建筑
                logger.log("放置植被与建筑...\n")
                task_id = perf.start("vegetation_buildings")
                
                try:
                    vegetation, buildings, roads, settlements = place_vegetation_and_buildings(
                        biome_map, 
                        height_map, 
                        rivers, 
                        preferences, 
                        map_params
                    )
                except Exception as e:
                    logger.log(f"植被和建筑生成错误: {e}", "ERROR")
                    # 提供默认值避免后续过程崩溃
                    vegetation, buildings, roads, settlements = [], [], [], []
                
                perf.end(task_id)
                
                # 存储对象层
                map_data.add_object_layer("vegetation", vegetation)
                map_data.add_object_layer("buildings", buildings)
                map_data.add_object_layer("settlements", settlements)
                map_data.add_object_layer("roads", roads)
                
                # 在建筑和植被可视化前添加
                logger.log(f"可视化前检查: vegetation={len(vegetation)}, buildings={len(buildings)}, roads={len(roads)}")
                # 可视化：植被和建筑
                if visualize:
                    vis.visualize_objects(height_map, vegetation, buildings, roads, "植被与建筑", "对象放置")
                
                # 内容布局和道路生成
                logger.log("NEAT进化内容布局...\n")
                task_id = perf.start("content_layout")
                
                try:
                    order, content_layout = run_neat_for_content_layout(buildings)
                except Exception as e:
                    logger.log(f"内容布局生成错误: {e}", "ERROR")
                    order, content_layout = [], {}
                
                perf.end(task_id)
                
                logger.log(f"道路生成参数检查: buildings={len(buildings)}, order={len(order)}, rivers类型={type(rivers)}, rivers形状={rivers.shape if hasattr(rivers, 'shape') else 'N/A'}")
                if len(buildings) > 0:
                    logger.log(f"建筑示例: {buildings[0]}")
                if settlements:
                    logger.log(f"聚居点数量: {len(settlements)}")
                
                # 构建道路网络
                logger.log("构建智能道路网络...\n")
                task_id = perf.start("road_generation")
                
                try:
                    roads_network, roads_types = build_roads(
                        height_map, 
                        buildings, 
                        order,
                        rivers,
                        biome_map,
                        settlements
                    )
                except Exception as e:
                    logger.log(f"道路网络生成错误: {e}", "ERROR")
                    # 提供默认数组避免后续过程崩溃
                    roads_network = np.zeros((height, width), dtype=np.uint8)
                    roads_types = np.zeros((height, width), dtype=np.uint8)
                
                # 确保道路地图是数组
                roads_map = np.array(roads_network, dtype=np.uint8)

                # 创建并填充道路图层
                if "roads_map" not in map_data.layers:
                    map_data.create_layer("roads_map", dtype=np.uint8)

                # 添加类型检查与转换
                try:
                    # 检查是否是GPU数组
                    if hasattr(map_data.layers["roads_map"], 'get') and hasattr(map_data.layers["roads_map"], 'device'):
                        # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                        logger.log("道路图层：将NumPy数组转换为CuPy数组")
                        import cupy as cp
                        cp_roads_map = cp.asarray(roads_map, dtype=cp.uint8)
                        map_data.layers["roads_map"][:] = cp_roads_map
                    else:
                        # 如果是CPU数组(NumPy)，直接赋值
                        logger.log("道路图层：使用NumPy数组直接赋值")
                        map_data.layers["roads_map"][:] = roads_map
                except Exception as e:
                    logger.log(f"道路图层数据赋值失败: {e}", "ERROR")
                    # 降级方案：逐个元素复制
                    try:
                        h, w = roads_map.shape
                        for y in range(h):
                            for x in range(w):
                                map_data.layers["roads_map"][y, x] = int(roads_map[y, x])
                        logger.log("使用逐元素复制方法完成道路图层赋值", "WARNING")
                    except Exception as e2:
                        logger.log(f"道路图层降级方案也失败: {e2}", "ERROR")
                        raise

                # 创建并填充道路类型图层 - 同样需要类型检查
                if "roads_types" not in map_data.layers:
                    map_data.create_layer("roads_types", dtype=np.uint8)

                # 添加类型检查与转换
                try:
                    # 检查是否是GPU数组
                    if hasattr(map_data.layers["roads_types"], 'get') and hasattr(map_data.layers["roads_types"], 'device'):
                        # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                        logger.log("道路类型图层：将NumPy数组转换为CuPy数组")
                        import cupy as cp
                        cp_roads_types = cp.asarray(roads_types, dtype=cp.uint8)
                        map_data.layers["roads_types"][:] = cp_roads_types
                    else:
                        # 如果是CPU数组(NumPy)，直接赋值
                        logger.log("道路类型图层：使用NumPy数组直接赋值")
                        map_data.layers["roads_types"][:] = np.array(roads_types, dtype=np.uint8)
                except Exception as e:
                    logger.log(f"道路类型图层数据赋值失败: {e}", "ERROR")
                    # 降级方案：逐个元素复制
                    try:
                        h, w = roads_types.shape
                        for y in range(h):
                            for x in range(w):
                                map_data.layers["roads_types"][y, x] = int(roads_types[y, x])
                        logger.log("使用逐元素复制方法完成道路类型图层赋值", "WARNING")
                    except Exception as e2:
                        logger.log(f"道路类型图层降级方案也失败: {e2}", "ERROR")
                        raise
                
                perf.end(task_id)

                # 在道路网络可视化前添加
                if 'road_points' in locals():
                    logger.log(f"道路点数量: {len(road_points)}, roads_map非零元素: {np.count_nonzero(roads_map)}")
                else:
                    logger.log(f"roads_map非零元素: {np.count_nonzero(roads_map)}")
                # 可视化：道路网络
                if visualize:
                    road_points = []
                    if roads_map is not None:
                        for y in range(height):
                            for x in range(width):
                                if roads_map[y, x] > 0:
                                    road_points.append({"x": x, "y": y})
                    # 添加检查确保有足够数据显示
                    if len(road_points) > 0:
                        vis.visualize_objects(height_map, None, None, road_points, "道路网络", "交通系统")
                    else:
                        logger.log("警告: 道路网络为空，跳过可视化", "WARNING")

                # 添加LLM内容与故事生成阶段
                logger.log("LLM处理内容与故事生成...\n")
                task_id = perf.start("llm_processing")

                try:
                    if not content_layout:
                        content_layout = {"objects": []}
                    elif "objects" not in content_layout:
                        content_layout["objects"] = []

                    # 调用LLM处理任务
                    content_layout = llm.process_llm_tasks(
                        content_layout,
                        map_data.layers["biome"],
                        preferences
                    )

                    # 将LLM生成的内容添加到地图数据中
                    if "story_events" in content_layout:
                        valid_events = []
                        for i, event in enumerate(content_layout["story_events"]):
                            if not isinstance(event, dict):
                                logger.log(f"故事事件索引 {i} 不是字典类型: {event}", "ERROR")
                                continue
                            if "x" not in event or "y" not in event:
                                logger.log(f"故事事件索引 {i} 缺少x或y属性: {event}", "ERROR")
                                continue
                            try:
                                event["x"] = int(event["x"])
                                event["y"] = int(event["y"])
                                valid_events.append(event)
                                logger.log(f"验证通过的故事事件: x={event['x']}, y={event['y']}", "DEBUG")
                            except (ValueError, TypeError):
                                logger.log(f"故事事件索引 {i} 的坐标无效: {event}", "WARNING")

                        if valid_events:
                            # 调用时使用skip_invalid参数
                            map_data.add_object_layer("story_events", valid_events, skip_invalid=True)
                            logger.log(f"LLM生成了{len(valid_events)}个故事事件点")
                            
                            # 生成扩展的故事剧情内容
                            try:
                                logger.log("开始生成扩展的游戏剧情...")
                                expanded_story = llm.generate_story_expansion(
                                    valid_events,
                                    map_data.layers["biome"],
                                    preferences
                                )
                                map_data.story_content = expanded_story
                                logger.log("完成游戏剧情扩展，生成了完整故事线")
                            except Exception as e:
                                logger.log(f"故事剧情扩展生成错误: {e}", "ERROR")
                        else:
                            logger.log("没有有效的故事事件点可添加", "WARNING")

                except Exception as e:
                    logger.log(f"LLM内容生成错误: {e}", "ERROR")
                    if not content_layout:
                        content_layout = {"objects": [], "story_events": []}

                perf.end(task_id)
                
                # 交互式进化 - 同样需要避免重复执行
                logger.log("IEC(交互评价)...\n")
                task_id = perf.start("interactive_evolution")

                try:
                    # 初始化进化引擎
                    engine = BiomeEvolutionEngine(
                        biome_map=biome_map,
                        height_map=height_map,
                        temperature_map=temp_map,
                        moisture_map=humid_map,
                        memory_path="./data/models/evolution_memory.dat"
                    )
                    
                    # 从map_params获取进化代数，不存在则默认为1
                    generations = map_params.get("evolution_generations", 1)
                    logger.log(f"将进行 {generations} 代进化...")
                    
                    # 检查是否支持交互式模式
                    is_interactive = map_params.get("interactive_evolution", True)
                    
                    # 已完成的代数（用于恢复状态）
                    completed_gens = 0
                    if hasattr(map_data, 'generation_state') and map_data.generation_state:
                        completed_gens = map_data.generation_state.get('completed_gens', 0)
                        logger.log(f"从第 {completed_gens+1} 代恢复进化...")
                    
                    for gen in range(completed_gens, generations):
                        logger.log(f"第 {gen+1} 代进化...")
                        
                        # 获取用户评分
                        if is_interactive and not goto_post_evolution:  # 如果是交互模式且不是从进化后恢复
                            if parent_frame and use_gui_editors:
                                # GUI集成模式 - 保存状态并暂停执行
                                logger.log("等待GUI界面完成评分...")
                                
                                # 保存当前状态到map_data中
                                map_data.pending_editor = "evolution_scorer"
                                map_data.editor_state = {
                                    'engine': engine,
                                    'generations': generations,
                                    'map_params': map_params
                                }
                                map_data.generation_state = {
                                    'resume_point': 'post_evolution',
                                    'width': width,
                                    'height': height,
                                    'completed_gens': gen,
                                    'map_params': map_params
                                }
                                
                                # 确保标记为未完成
                                map_data.generation_complete = False
                                
                                # 返回当前状态，中断执行
                                return map_data
                            else:
                                # 独立窗口模式
                                try:
                                    # 打开评分界面并等待用户输入
                                    logger.log("正在打开独立评分界面，请在界面中选择您喜欢的方案...")
                                    user_scores = get_visual_scores(engine)
                                    logger.log(f"用户评分已完成: {len(user_scores)} 个方案被评分")
                                except Exception as score_error:
                                    logger.log(f"评分界面出错: {score_error}，使用自动评分", "WARNING")
                                    # 降级为自动评分
                                    user_scores = [np.random.uniform(5, 8) for _ in range(engine.population_size)]
                        else:
                            # 以下情况使用自动评分:
                            # 1. 非交互模式
                            # 2. 从进化后恢复状态
                            # 3. 评分出错降级
                            logger.log("使用自动评分模式...")
                            user_scores = [np.random.uniform(5, 8) for _ in range(engine.population_size)]
                            
                            # 如果是从进化点恢复，需要跳过，因为进化已经完成
                            if goto_post_evolution and gen == completed_gens:
                                logger.log("从进化后恢复状态，跳过首次进化...")
                                continue
                        
                        # 确保评分列表长度与种群大小匹配
                        if len(user_scores) != engine.population_size:
                            logger.log(f"评分数量({len(user_scores)})与种群大小({engine.population_size})不匹配，调整中...", "WARNING")
                            # 如果评分数量不够，补充随机评分
                            while len(user_scores) < engine.population_size:
                                user_scores.append(np.random.uniform(5, 7))
                            # 如果评分数量过多，截断
                            if len(user_scores) > engine.population_size:
                                user_scores = user_scores[:engine.population_size]
                        
                        # 执行进化
                        logger.log(f"基于评分进行第 {gen+1} 代进化...")
                        engine.evolve_generation(user_scores)
                        logger.log(f"第 {gen+1} 代进化完成")
                        
                        # 更新生成状态中的已完成代数
                        if hasattr(map_data, 'generation_state'):
                            map_data.generation_state['completed_gens'] = gen + 1
                    
                    # 获取进化后的最佳地图
                    evolved_biome_map = engine.best_individual
                    
                    # 验证并应用进化结果
                    if isinstance(evolved_biome_map, np.ndarray) and evolved_biome_map.shape == (height, width):
                        try:
                            # 尝试直接赋值
                            map_data.layers["biome"][:] = evolved_biome_map
                            # 更新本地变量
                            biome_map = evolved_biome_map
                        except ValueError as e:
                            logger.log(f"数组类型不兼容错误: {e}", "WARNING")
                            # 逐元素拷贝
                            h, w = evolved_biome_map.shape
                            for y in range(h):
                                for x in range(w):
                                    map_data.layers["biome"][y, x] = int(evolved_biome_map[y, x])
                        
                        logger.log("已应用进化后的生物群落地图")
                    else:
                        logger.log("生物群落进化结果格式无效，保留原始数据", "WARNING")
                except Exception as e:
                    logger.log(f"交互式进化错误: {e}", "ERROR")
                    import traceback
                    logger.log(traceback.format_exc(), "DEBUG")

                perf.end(task_id)
                
                # 检查biome_map是否需要从map_data中重新获取
                if goto_post_evolution:
                    biome_map = map_data.get_layer("biome")
                    logger.log("从map_data获取进化后的生物群系地图")
                
                """
                # 计算情感信息
                logger.log("计算情感信息...\n")
                task_id = perf.start("emotion_calculation")
                
                try:
                    # 创建分析器实例
                    analyzer = MapEmotionAnalyzer(
                        map_data.get_layer("height"), 
                        biome_map, 
                        vegetation, 
                        buildings, 
                        rivers, 
                        content_layout, 
                        cave_entrances,
                        roads,
                        roads_map
                    )
                    
                    # 分析情感信息
                    content_layout = analyzer.analyze_map_emotions(
                        biome_map, vegetation, buildings, rivers, 
                        content_layout, cave_entrances, roads, roads_map
                    )
                    
                    # 保存情感信息到地图数据
                    map_data.content_layout = content_layout
                except Exception as e:
                    logger.log(f"情感分析错误: {e}", "ERROR")
                
                perf.end(task_id)
                """
                
                # 记录性能报告
                logger.log("生成性能报告...\n")
                performance_report = perf.report()
                logger.log(performance_report)
                logger.log("地图生成完成！\n")
                
                # 标记地图生成完成
                map_data.generation_complete = True
                # 清除生成状态和编辑器状态
                if hasattr(map_data, 'generation_state'):
                    map_data.generation_state = {'resume_point': 'complete'}
                if hasattr(map_data, 'pending_editor'):
                    map_data.pending_editor = None
                if hasattr(map_data, 'editor_state'):
                    map_data.editor_state = None
                
                # 确保GPU数据转回CPU进行返回
                if config.use_gpu:
                    map_data.to_cpu()
                
                return map_data
            except Exception as e:
                logger.log(f"线程池执行错误: {e}", "ERROR")
                raise            
    except Exception as e:
        # 捕获整个生成过程中的任何未处理异常
        logger.log(f"地图生成出现严重错误: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "DEBUG")
        raise
  
        #返回与原始函数相同的结构，确保兼容性
        #废案：return (map_data.get_layer("height"), biome_map, vegetation, buildings, rivers, content_layout, caves, map_params, biome_data, roads,roads_map)


from core.generation_steps import StepManager
from core.step_registry import register_all_steps

def generate_map_modular(preferences, width, height, export_model=False, logger=None, 
                 use_real_geo_data=False, geo_data_path=None, geo_bounds=None,
                 visualize=False, visualization_path=None, 
                 parent_frame=None, use_gui_editors=False,
                 resume_state=None, callback=None,
                 enabled_steps=None, step_order=None):
    """模块化地图生成函数，使用可配置的步骤系统
    
    新增参数:
        enabled_steps: 启用的步骤ID列表，如果为None则使用默认设置
        step_order: 步骤执行顺序，如果为None则使用默认顺序
    """
    if logger is None:
        # 如果未提供 logger，则使用默认配置
        logger.log("警告未提供logger")
        logger = ThreadSafeLogger(max_lines=500, log_file="map_generation.log")
    
    # 初始化步骤管理器
    step_manager = StepManager()
    register_all_steps(step_manager)
    
    # 如果提供了自定义步骤配置，应用它们
    if enabled_steps is not None:
        # 首先禁用所有步骤
        for step in step_manager.get_all_steps():
            step.enabled = False
        # 然后启用指定的步骤
        for step_id in enabled_steps:
            step = step_manager.get_step(step_id)
            if step:
                step.enabled = True
            else:
                logger.log(f"警告：未找到步骤 {step_id}", "WARNING")
    
    # 如果提供了自定义步骤顺序，应用它
    if step_order is not None:
        try:
            step_manager.set_step_order(step_order)
        except ValueError as e:
            logger.log(f"设置步骤顺序失败: {str(e)}", "ERROR")
            return None
    
    # 验证依赖关系
    errors = step_manager.validate_dependencies()
    if errors:
        error_msg = "\n".join(errors)
        logger.log(f"步骤依赖关系验证失败:\n{error_msg}", "ERROR")
        return None
    
    # 初始化性能监控和配置
    perf = PerformanceMonitor()
    config = MapGenerationConfig(width=width, height=height, export_model=export_model)
    
    # 准备初始上下文
    context = {
        "preferences": preferences,
        "width": width,
        "height": height,
        "export_model": export_model,
        "use_real_geo_data": use_real_geo_data,
        "geo_data_path": geo_data_path,
        "geo_bounds": geo_bounds,
        "visualize": visualize,
        "visualization_path": visualization_path,
        "parent_frame": parent_frame,
        "use_gui_editors": use_gui_editors,
        "resume_state": resume_state,
        "callback": callback,
        "perf": perf,
        "config": config,
        "logger": logger
    }
    
    # 初始化可视化工具（如果启用）
    if visualize:
        from core.services.map_tools import MapGenerationVisualizer
        vis = MapGenerationVisualizer(save_path=visualization_path, show_plots=True, logger=logger, gui_frame=visualize)
        logger.log("已启用地图生成可视化功能")
        context["vis"] = vis
    
    try:
        # 执行所有步骤
        logger.log("开始按步骤生成地图...")
        context = step_manager.execute_steps(context, logger)
        
        # 创建并返回最终地图数据
        map_data = create_map_data_from_context(context, config)
        logger.log("地图生成完成！")
        
        return map_data
    except Exception as e:
        logger.log(f"地图生成出现严重错误: {str(e)}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
        return None

def create_map_data_from_context(context, config):
    """从步骤执行上下文创建最终的地图数据对象"""
    width = context.get("width")
    height = context.get("height")
    map_data = MapData(width, height, config)
    
    # 将上下文中的数据复制到map_data中
    # 高度图
    height_map = context.get("updated_height_map") or context.get("carved_height_map") or context.get("edited_height_map") or context.get("height_map")
    if height_map is not None:
        map_data.layers["height"] = height_map
    
    # 温度图
    temp_map = context.get("final_temp_map") or context.get("enhanced_temp_map") or context.get("updated_temp_map") or context.get("temperature_map")
    if temp_map is not None:
        map_data.layers["temperature"] = temp_map
    
    # 湿度图
    humid_map = context.get("final_humid_map") or context.get("enhanced_humid_map") or context.get("updated_humid_map") or context.get("humidity_map")
    if humid_map is not None:
        map_data.layers["humidity"] = humid_map
    
    # 生物群系图
    biome_map = context.get("evolved_biome_map") or context.get("transition_biome_map") or context.get("biome_map")
    if biome_map is not None:
        if "biome" not in map_data.layers:
            map_data.create_layer("biome", dtype=np.int32, fill_value=0)
        map_data.layers["biome"][:] = biome_map
    
    # 河流图
    river_map = context.get("river_map")
    if river_map is not None:
        map_data.layers["rivers"] = river_map
    
    # 河流特征
    river_features = context.get("river_features")
    if river_features is not None:
        map_data.layers["river_features"] = river_features
    
    # 河流点
    river_points = context.get("river_points")
    if river_points is not None:
        map_data.layers["rivers_points"] = river_points
    
    # 洞穴数据
    cave_data = context.get("cave_data")
    if cave_data is not None:
        map_data.layers["caves"] = cave_data.get("entrances", [])
    
    # 微气候
    microclimate_map = context.get("microclimate_map")
    if microclimate_map is not None:
        map_data.layers["microclimate"] = microclimate_map
    
    # 植被和建筑
    vegetation = context.get("vegetation")
    if vegetation is not None:
        map_data.add_object_layer("vegetation", vegetation)
    
    buildings = context.get("buildings")
    if buildings is not None:
        map_data.add_object_layer("buildings", buildings)
    
    settlements = context.get("settlements")
    if settlements is not None:
        map_data.add_object_layer("settlements", settlements)
    
    roads = context.get("roads")
    if roads is not None:
        map_data.add_object_layer("roads", roads)
    
    # 道路网络
    roads_network = context.get("roads_network")
    if roads_network is not None:
        if "roads_map" not in map_data.layers:
            map_data.create_layer("roads_map", dtype=np.uint8)
        map_data.layers["roads_map"][:] = roads_network
    
    # 道路类型
    roads_types = context.get("roads_types")
    if roads_types is not None:
        if "roads_types" not in map_data.layers:
            map_data.create_layer("roads_types", dtype=np.uint8)
        map_data.layers["roads_types"][:] = roads_types
    
    # 内容布局
    content_layout = context.get("final_content_layout") or context.get("enhanced_content_layout") or context.get("content_layout")
    if content_layout is not None:
        map_data.content_layout = content_layout
    
    # 故事事件
    story_events = context.get("story_events")
    if story_events:
        map_data.add_object_layer("story_events", story_events, skip_invalid=True)
    
    # 设置生成完成标志
    map_data.generation_complete = True
    
    return map_data

"""关于保持rivers为数组的必要性
不能将rivers简单修改为单个布尔值，因为：
地图需要知道每个位置是否有河流，这需要一个与地图大小相同的数组
用单个布尔值只能表示"地图上有/没有河流"的整体信息，无法表示河流的具体分布""" 
