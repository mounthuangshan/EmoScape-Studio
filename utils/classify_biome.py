#from __future__ import annotations
#标准库
import logging

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#网络与并发

#其他工具
from dataclasses import dataclass, asdict, field
#import logging

###############################
#根据环境参数（如高度、温度、湿度等）将地图区域分类为不同生物群系（如森林、沙漠、草原等）
###############################
def classify_biome(height_map, temp_map, humid_map, biome_data):
    """返回形状与输入一致的NumPy数组，使用整数编码生物群系"""

    # 输入数据验证
    assert height_map.shape == temp_map.shape == humid_map.shape, "所有输入矩阵维度必须一致"
    assert isinstance(height_map, np.ndarray), "输入必须为NumPy数组"
    
    # 创建默认生物群系模板
    DEFAULT_BIOME_ID = 0
    REQUIRED_FIELDS = ['name', 'height_condition', 'temp_range', 'hum_range']
    
    # 预处理生物群系数据
    validated_biomes = []
    for biome in biome_data["biomes"]:
        # 验证必要字段
        missing = [f for f in REQUIRED_FIELDS if f not in biome]
        if missing:
            logging.error(f"生物群系{biome.get('name', '未知')}缺失必要字段{missing}，已跳过")
            continue
        
        # 转换高度条件为可执行格式
        try:
            cond = biome['height_condition']
            biome['_condition'] = (
                cond[0],  # 操作符
                biome_data['sea_level'] if cond[1] == 'sea_level' else float(cond[1])
            )
        except:
            logging.error(f"生物群系{biome.get('name', '未知')}高度条件解析失败，已跳过")
            continue
        
        validated_biomes.append(biome)

    # 创建输出数组 - 默认全部为0而不是DEFAULT_BIOME_ID
    h, w = height_map.shape
    biome_ids = np.zeros((h, w), dtype=np.int32)  # 使用int32与预览代码保持一致
    
    # 向量化计算提升性能
    sea_level = biome_data.get('sea_level', 0.5)
    
    # 对每个生物群系生成掩码 - 保持索引与配置文件一致
    for biome_idx, biome in enumerate(validated_biomes):
        # 使用配置文件的原始索引
        biome_id = biome_idx
        
        # 解析条件
        op, ref_val = biome['_condition']
        t_min, t_max = biome['temp_range']
        h_min, h_max = biome['hum_range']
        
        # 生成条件掩码
        if op == '<':
            height_mask = height_map < ref_val
        elif op == '>=':
            height_mask = height_map >= ref_val
        else:
            continue
        
        temp_mask = (temp_map >= t_min) & (temp_map <= t_max)
        hum_mask = (humid_map >= h_min) & (humid_map <= h_max)
        
        # 组合掩码
        full_mask = height_mask & temp_mask & hum_mask
        
        # 更新满足条件的区域
        biome_ids[full_mask] = biome_id
    
    # 检查是否有超出范围的ID
    max_valid_id = len(validated_biomes) - 1
    invalid_ids = np.unique(biome_ids[biome_ids > max_valid_id])
    if len(invalid_ids) > 0:
        logging.warning(f"发现超出有效范围的生物群系ID: {invalid_ids}")
        # 将超出范围的ID重置为0
        biome_ids[biome_ids > max_valid_id] = 0
        
    return biome_ids

def match_biome(biome_data, H, T, Hum, sea_level):
    """辅助函数，供需要逐点查询时使用"""
    for biome_idx, biome in enumerate(biome_data["biomes"]):
        # 使用与配置文件一致的索引
        biome_id = biome_idx
        
        # 解析高度条件
        cond = biome['height_condition']
        op, ref = cond[0], sea_level if cond[1] == 'sea_level' else float(cond[1])
        
        # 检查高度
        if op == '<' and not (H < ref):
            continue
        if op == '>=' and not (H >= ref):
            continue
        
        # 检查温湿度
        t_min, t_max = biome['temp_range']
        h_min, h_max = biome['hum_range']
        if (T >= t_min) and (T <= t_max) and (Hum >= h_min) and (Hum <= h_max):
            return biome_id
    
    return 0  # 返回默认ID