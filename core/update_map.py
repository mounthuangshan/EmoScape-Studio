#from __future__ import annotations
#标准库
import os
import json

#数据处理与科学计算

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发

#其他工具
import logging
import os
import json

#项目文件
from utils.llm import *
from utils.tools import *

############
#更新地图
############
def update_map(map_data, gui_log=None):
    """
    更新地图数据，使用player_feedback.json中的反馈调整
    
    参数:
        map_data: MapData对象，包含所有地图层
        gui_log: 可选的GUI日志输出对象
    
    返回:
        更新后的MapData对象
    """
    # 从MapData对象中解包数据
    height_map = map_data.get_layer("height")
    biome_map = map_data.get_layer("biome")
    vegetation = map_data.layers.get("vegetation", [])
    buildings = map_data.layers.get("buildings", [])
    rivers = map_data.get_layer("rivers")
    content_layout = map_data.content_layout
    caves = map_data.get_layer("caves")
    map_params = map_data.params
    biome_data = map_data.biome_data
    roads = map_data.layers.get("roads", [])
    roads_map = map_data.get_layer("roads_map")

    if gui_log: gui_log.insert("end", "检查玩家反馈...\n")
    feedback = None
    if os.path.exists("player_feedback.json"):
        with open("player_feedback.json", "r", encoding="utf-8") as f:
            feedback = json.load(f)
    
    if feedback is not None:
        if gui_log: gui_log.insert("end", "玩家反馈已加载，更新偏好与地图...\n")
        new_prefs = {
            "achievement": feedback.get("achievement", 0.5),
            "exploration": feedback.get("exploration", 0.5),
            "social": feedback.get("social", 0.5),
            "combat": feedback.get("combat", 0.5)
        }
        map_params = map_preferences_to_parameters(new_prefs)
        content_layout = refine_layout_with_llm(content_layout)
        content_layout = enrich_story_with_llm(content_layout, biome_map, new_prefs)
        if gui_log: gui_log.insert("end", "根据反馈更新完成...\n")

    # 更新MapData对象
    map_data.layers["height"] = height_map
    map_data.layers["biome"] = biome_map
    map_data.layers["vegetation"] = vegetation
    map_data.layers["buildings"] = buildings
    map_data.layers["rivers"] = rivers
    map_data.content_layout = content_layout
    map_data.layers["caves"] = caves
    map_data.params = map_params
    map_data.biome_data = biome_data
    map_data.layers["roads"] = roads
    map_data.layers["roads_map"] = roads_map

    return map_data