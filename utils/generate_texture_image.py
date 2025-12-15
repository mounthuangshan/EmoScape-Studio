#from __future__ import annotations
#标准库

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

###########################################
#利用 PIL 库生成地图纹理图像，供 3D 模型使用。
###########################################
def safe_cell_to_bool(cell):
    """安全转换任意类型为布尔值"""
    if isinstance(cell, np.ndarray):
        return bool(np.any(cell))
    return bool(cell)

def generate_texture_image(biome_map, caves, rivers, vegetation, buildings, roads, content_layout):

    # 获取地图尺寸
    h = len(biome_map)
    w = len(biome_map[0]) if h > 0 else 0
    # 创建空集
    cave_set = set()
    
    # 创建道路坐标集合（兼容不同数据结构）
    road_set = set()
    # 创建道路坐标集合（兼容不同数据结构）
    road_set = set()

    # 遍历元组
    for item in caves:
        if isinstance(item, np.ndarray):
            # 将 NumPy 数组元素转换为元组并添加到集合
            cave_set.update(map(tuple, item.reshape(-1, 1)))
        elif isinstance(item, list):
            # 将列表元素转换为元组并添加到集合
            cave_set.update(map(tuple, item))
    veg_set=set((v[0],v[1]) for v in vegetation)
    bld_set=set((b[0],b[1]) for b in buildings)
    story_set=set()
    if "story_events" in content_layout:
        story_set=set((e["x"],e["y"]) for e in content_layout["story_events"])
        
    creature_set=[]
    if content_layout and "creatures" in content_layout:
        for cr in content_layout["creatures"]:
            creature_set.append((cr["x"],cr["y"]))

    # 强制转换为列表的列表
    rivers = [list(row) for row in rivers]
    
    # 处理numpy数组输入
    if isinstance(rivers, np.ndarray):
        if rivers.ndim > 2:
            rivers = np.any(rivers, axis=2)  # 合并多余维度
        rivers = rivers.tolist()

    # 转换为标准二维布尔列表
    rivers = [
        [
            safe_cell_to_bool(cell)
            for cell in list(row)[:w]  # 截断列
        ] + [False]*(w - len(list(row)[:w]))  # 填充列
        for row in 
        list(rivers)[:h] +  # 截断行
        [[False]*w]*(h - len(list(rivers)[:h]))  # 填充行
    ]

    # ========== 验证维度 ==========
    assert len(rivers) == h, f"河流行数错误: {len(rivers)} vs {h}"
    assert all(len(row) == w for row in rivers), f"河流列数错误: {len(rivers[0])} vs {w}"

    # 填充缺失部分
    rivers += [[False]*w for _ in range(h - len(rivers))]

    # 验证河流数据
    assert len(rivers) == h and all(len(row)==w for row in rivers), \
        f"河流数据维度异常：{len(rivers)}行 vs {h}行，首行{len(rivers[0])}列 vs {w}列"

    img=Image.new("RGB",(w,h))
    pixels=img.load()
    for j in range(h):
        for i in range(w):
            biome=biome_map[j][i]
            base_c=biome.get("color",[0.5,0.5,0.5])
            c=base_c
            if rivers[j][i]:
                c=(0,0.5,1)
            if (i,j) in cave_set:
                c=(0,0,0)
            if (i,j) in veg_set:
                c=(0,1,0)
            if (i, j) in road_set:  # 修改为集合判断
                c = (0.6, 0.6, 0.6)
            if (i,j) in story_set:
                c=(1,0,1)
            if (i,j) in bld_set:
                c=(1,0,0)
            r=int(c[0]*255)
            g=int(c[1]*255)
            b=int(c[2]*255)
            pixels[i,j]=(r,g,b)
    img.save("terrain_texture.png")


def deep_convert_to_bool(data):
    """处理嵌套结构但保留维度"""
    if isinstance(data, np.ndarray):
        # 保留数组维度，仅转换元素
        return [deep_convert_to_bool(item) for item in data.tolist()]
    elif isinstance(data, (list, tuple)):
        # 保留列表/元组结构
        return [deep_convert_to_bool(item) for item in data]
    else:
        # 标量直接转换
        return bool(data)