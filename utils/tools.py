#from __future__ import annotations
#标准库
import random
import math
import hashlib
import time
import numpy as np
from math import floor
from typing import Tuple
import json

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



##################
#数据加载与偏好映射
##################
def load_objects_db(logger):
    """加载对象数据库并记录日志"""
    try:
        logger.log("开始加载对象数据库...")
        # 记录文件打开操作
        logger.log(f"正在打开 objects_db.json")
        with open("./data/configs/objects_db.json", "r", encoding="utf-8") as f:
            data = json.load(f) 
        # 处理数据
        all_objs = []
        for category, cat_objs in data.items():
            logger.log(f"正在处理分类 [{category}]，包含 {len(cat_objs)} 个对象")
            all_objs.extend(cat_objs)
            
        logger.log(f"数据库加载完成，总对象数：{len(all_objs)}")
        return data, all_objs
        
    # 必须至少有一个 except 或 finally 块
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise
    except Exception as e:  # 通用异常捕获必须放在最后
        logger.error(f"未知错误: {str(e)}")
        raise
    # 如果需要可以添加 finally 块
    # finally:
    #     logger.log("清理资源")

def load_biome_config(logger):
    try:
        # 记录文件打开操作
        logger.log(f"正在打开 biomes_config.json")
        with open("./data/configs/biomes_config.json","r",encoding="utf-8") as f:
            data=json.load(f)
        return data

    # 必须至少有一个 except 或 finally 块
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise
    except Exception as e:  # 通用异常捕获必须放在最后
        logger.error(f"未知错误: {str(e)}")

def load_player_feedback(logger):
    try:
        # 记录文件打开操作
        logger.log(f"正在打开 player_feedback.json")
        with open("./data/configs/player_feedback.json","r",encoding="utf-8") as f:
            data=json.load(f)
        return data
    
    # 必须至少有一个 except 或 finally 块
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise
    except Exception as e:  # 通用异常捕获必须放在最后
        logger.error(f"未知错误: {str(e)}")

def map_preferences_to_parameters(preferences):
    ach = preferences.get("achievement",0.5)
    exp = preferences.get("exploration",0.5)
    soc = preferences.get("social",0.5)
    cbt = preferences.get("combat",0.5)
    island_count = int(3 + exp*4)
    vegetation_coverage = 0.2 + exp*0.5
    water_area = 0.1 + ach*0.2
    human_coverage = 0.05 + soc*0.3
    city_count = int(1 + soc*4)
    # 添加进化代数参数
    evolution_generations = preferences.get("evolution_generations", 1)
    max_pref = max(ach,exp,soc,cbt)
    if max_pref==exp:
        city_layout="ring"
    elif max_pref==soc:
        city_layout="star"
    elif max_pref==cbt:
        city_layout="line"
    else:
        city_layout="line"
        
    # 创建基础参数字典
    params = {
        "island_count": island_count,
        "vegetation_coverage": vegetation_coverage,
        "water_area": water_area,
        "human_coverage": human_coverage,
        "city_count": city_count,
        "city_layout": city_layout,
        "evolution_generations": evolution_generations
    }
    
    # 添加地形参数，从preferences直接复制相关参数
    terrain_params = [
        "seed", "scale_factor", "mountain_sharpness", "erosion_iterations",
        "river_density", "use_tectonic", "detail_level", "use_frequency_optimization",
        "erosion_type", "erosion_strength", "talus_angle", "sediment_capacity",
        "rainfall", "evaporation", "min_watershed_size", "precipitation_factor",
        "meander_factor", "octaves", "persistence", "lacunarity", "plain_ratio",
        "hill_ratio", "mountain_ratio", "plateau_ratio", "latitude_effect",
        "prevailing_wind_x", "prevailing_wind_y"
    ]
    
    for param in terrain_params:
        if param in preferences:
            params[param] = preferences[param]
    
    return params
#######################################################################################
class Creature:
    latent_dim=16
    hidden_dim=16
    output_dim=8

    def __init__(self, weights=None, role=None):
        if weights is None:
            W1=self.rand_weight((self.latent_dim,self.hidden_dim))
            b1=np.zeros(self.hidden_dim)
            W2=self.rand_weight((self.hidden_dim,self.output_dim))
            b2=np.zeros(self.output_dim)
            self.weights=(W1,b1,W2,b2)
            self.latent=np.random.randn(self.latent_dim)
        else:
            W1,b1,W2,b2,latent=weights
            self.weights=(W1.copy(),b1.copy(),W2.copy(),b2.copy())
            self.latent=latent.copy()

        if role is None:
            self.role = random.choice(["Predator","Prey"])
        else:
            self.role = role

    @staticmethod
    def rand_weight(shape):
        return np.random.randn(*shape)*0.1

    def copy(self):
        W1,b1,W2,b2=self.weights
        return Creature(weights=(W1,b1,W2,b2,self.latent),role=self.role)

    def forward(self):
        W1,b1,W2,b2=self.weights
        x=self.latent
        h=np.dot(x,W1)+b1
        h=np.maximum(h,0) # ReLU
        out=np.dot(h,W2)+b2
        attrs={}
        for i,attr in enumerate(ATTR_NAMES):
            mini,maxi=ATTR_RANGES[attr]
            val=(1/(1+math.exp(-out[i])))
            val=mini+val*(maxi-mini)
            attrs[attr]=val
        return attrs

    def difficulty(self):
        c=self.forward()
        diff=(c["MoveSpeed"] 
              + c["Range"]/2 
              + c["PhysAttack"] 
              + c["PhysDefense"] 
              + c["AttackSpeed"]*2 
              + c["MagAttack"] 
              + c["MagDefense"] 
              - c["RespawnTime"]/10)
        return diff

    def mutate(self, mutation_rate=0.1):
        W1,b1,W2,b2=self.weights
        def mutate_arr(a):
            mask=(np.random.rand(*a.shape)<mutation_rate)
            a[mask]+=np.random.randn(np.count_nonzero(mask))*0.1
        mutate_arr(W1)
        mutate_arr(b1)
        mutate_arr(W2)
        mutate_arr(b2)
        self.weights=(W1,b1,W2,b2)
        if random.random()<mutation_rate:
            idx=random.randint(0,self.latent_dim-1)
            self.latent[idx]+=random.gauss(0,0.1)

    @staticmethod
    def crossover(p1,p2):
        W1_1,b1_1,W2_1,b2_1=p1.weights
        W1_2,b1_2,W2_2,b2_2=p2.weights
        def cross_arrays(a1,a2):
            shape=a1.shape
            a1f=a1.flatten()
            a2f=a2.flatten()
            cp=random.randint(1,len(a1f)-1)
            childf=np.concatenate([a1f[:cp],a2f[cp:]])
            return childf.reshape(shape)
        W1=cross_arrays(W1_1,W1_2)
        b1=cross_arrays(b1_1,b1_2)
        W2=cross_arrays(W2_1,W2_2)
        b2=cross_arrays(b2_1,b2_2)
        cp2=random.randint(1,p1.latent_dim-1)
        latent=np.concatenate([p1.latent[:cp2],p2.latent[cp2:]])
        role=random.choice([p1.role,p2.role])
        return Creature(weights=(W1,b1,W2,b2,latent),role=role)

################
#辅助函数
################
def get_env_story_attrs(height_map):
    avg_height=np.mean(height_map)
    mid_vals=[]
    for attr in ATTR_NAMES:
        mini,maxi=ATTR_RANGES[attr]
        mid=(mini+maxi)/2
        mid_vals.append(mid)
    env_story=mid_vals[:]
    env_story[0]=1+(avg_height/50)*9 # MoveSpeed
    env_story[3]=1+(avg_height/50)*49 # PhysDefense
    env_story[6]=40 # MagAttack期望值
    return env_story

def get_player_model_attrs(weak_vs_magic):
    mid_vals=[]
    for attr in ATTR_NAMES:
        mini,maxi=ATTR_RANGES[attr]
        mid=(mini+maxi)/2
        mid_vals.append(mid)
    if weak_vs_magic:
        mid_vals[6]=10
    return mid_vals

###################
#目标函数
###################
def objective_functions(creature, target_min, target_max, preferred_attrs, population=None, env_story_attrs=None, player_model_attrs=None):
    mid=(target_min+target_max)/2
    diff=creature.difficulty()
    obj1 = abs(diff - mid)

    attrs=creature.forward()
    vals=[attrs[a] for a in ATTR_NAMES]
    obj2=math.sqrt(sum((vals[i]-preferred_attrs[i])**2 for i in range(8)))

    if population is not None:
        predators=sum(1 for c in population if c.role=="Predator")
        preys=sum(1 for c in population if c.role=="Prey")
        if predators==0:
            obj3=100.0
        else:
            actual_ratio=preys/predators
            obj3=abs(actual_ratio - TARGET_RATIO)
    else:
        obj3=0.0

    if env_story_attrs is None:
        env_story_attrs=[(mini+maxi)/2 for (mini,maxi) in ATTR_RANGES.values()]
    obj4=math.sqrt(sum((vals[i]-env_story_attrs[i])**2 for i in range(8)))

    if player_model_attrs is None:
        player_model_attrs=[(mini+maxi)/2 for (mini,maxi) in ATTR_RANGES.values()]
    obj5=math.sqrt(sum((vals[i]-player_model_attrs[i])**2 for i in range(8)))

    return (obj1,obj2,obj3,obj4,obj5)

####################
#非支配排序与拥挤距离
####################
def dominates(a,b):
    better_in_all=True
    strictly_better=False
    for x,y in zip(a,b):
        if x>y:
            better_in_all=False
            break
        if x<y:
            strictly_better=True
    return better_in_all and strictly_better

def non_dominated_sort(pop, target_min, target_max, preferred_attrs, env_story_attrs, player_model_attrs):
    objs=[objective_functions(c,target_min,target_max,preferred_attrs,population=pop,env_story_attrs=env_story_attrs,player_model_attrs=player_model_attrs) for c in pop]
    S=[[] for _ in range(len(pop))]
    n=[0 for _ in range(len(pop))]
    front=[]
    for i in range(len(pop)):
        for j in range(len(pop)):
            if i!=j:
                if dominates(objs[i], objs[j]):
                    S[i].append(j)
                elif dominates(objs[j], objs[i]):
                    n[i]+=1
        if n[i]==0:
            front.append(i)
    fronts=[]
    current_front = front
    while current_front:
        next_front=[]
        for i in current_front:
            for j in S[i]:
                n[j]-=1
                if n[j]==0:
                    next_front.append(j)
        fronts.append(current_front)
        current_front=next_front
    return fronts, objs

def crowding_distance(front, objs):
    dist=[0.0 for _ in front]
    num_objectives = len(objs[0])
    for m in range(num_objectives):
        front_objs=[objs[i][m] for i in front]
        sorted_indices=sorted(range(len(front)), key=lambda idx: front_objs[idx])
        min_val=front_objs[sorted_indices[0]]
        max_val=front_objs[sorted_indices[-1]]
        dist[sorted_indices[0]]=float('inf')
        dist[sorted_indices[-1]]=float('inf')
        if max_val>min_val:
            for k in range(1,len(front)-1):
                dist[sorted_indices[k]] += (front_objs[sorted_indices[k+1]]-front_objs[sorted_indices[k-1]])/(max_val-min_val)
    return dist

##################
#选择、交叉、变异
##################
def selection(pop, target_min, target_max, preferred_attrs, env_story_attrs, player_model_attrs, pop_size):
    fronts, objs=non_dominated_sort(pop,target_min,target_max,preferred_attrs,env_story_attrs,player_model_attrs)
    new_pop=[]
    for front in fronts:
        if len(new_pop)+len(front)>pop_size:
            dist=crowding_distance(front,objs)
            order=sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
            for i in order:
                if len(new_pop)<pop_size:
                    new_pop.append(pop[front[i]])
                else:
                    break
            break
        else:
            for i in front:
                new_pop.append(pop[i])
    return new_pop,fronts,objs

def one_generation(pop, target_min, target_max, preferred_attrs, env_story_attrs, player_model_attrs, pop_size=50, mutation_rate=0.1):
    parents,_,_=selection(pop,target_min,target_max,preferred_attrs,env_story_attrs,player_model_attrs,pop_size)
    children=[]
    while len(children)<pop_size:
        p1=random.choice(parents)
        p2=random.choice(parents)
        child=Creature.crossover(p1,p2)
        child.mutate(mutation_rate)
        children.append(child)
    return children



######################
#地图初始化相关
#####################
def validate_seed(seed):
    """确保种子在int32范围内"""
    if seed is None:
        return np.random.randint(0, 999999)
        
    try:
        # 对大数取模以限制在合理范围内
        seed_int32 = seed % (2**31 - 1)
        return np.int32(seed_int32)
    except (OverflowError, ValueError, TypeError):
        # 如果转换失败，使用哈希值
        seed_str = str(seed)
        seed_hash = hash(seed_str) % (2**31 - 1)
        return np.int32(seed_hash)
    
def convert_tuple_to_dict(building_tuple: tuple) -> dict:
    """将元组转换为标准字典结构"""
    # 假设元组格式为 (type, width, height, [importance])
    return {
        'type': building_tuple[0] if len(building_tuple) > 0 else 'default',
        'width': building_tuple[1] if len(building_tuple) > 1 else 10,
        'height': building_tuple[2] if len(building_tuple) > 2 else 10,
        'importance': building_tuple[3] if len(building_tuple) > 3 else 5
    }
    
    
def generate_perlin_noise_2d(shape: Tuple[int, int], 
                             res: Tuple[int, int],
                             tileable: Tuple[bool, bool] = (False, False)) -> np.ndarray:
    """
    生成二维Perlin噪声图

    参数:
        shape: 输出噪声图的尺寸 (height, width)
        res: 每个维度的网格划分数量 (rows, cols)
        tileable: 是否生成可平铺的噪声 (x_tileable, y_tileable)

    返回:
        numpy.ndarray: 范围在[-1.0, 1.0]之间的噪声图
    """
    # 确保分辨率合理，避免太小导致索引错误
    res = (max(2, min(res[0], shape[0]//2)), 
           max(2, min(res[1], shape[1]//2)))
    
    # 生成梯度场的基础网格
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (max(1, shape[0] // res[0]), max(1, shape[1] // res[1]))
    
    # 生成随机梯度向量场
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    gradients = np.random.randn(res[0]+1, res[1]+1, 2)
    
    # 处理平铺模式
    if tileable[0]: gradients[-1,:] = gradients[0,:]
    if tileable[1]: gradients[:,-1] = gradients[:,0]
    
    # 噪声生成核心算法
    def perlin(pos):
        # 确定所在网格的四个角点
        x0, y0 = min(res[0]-1, floor(pos[0])), min(res[1]-1, floor(pos[1]))
        x1, y1 = min(res[0], x0 + 1), min(res[1], y0 + 1)
        fx, fy = pos[0] - x0, pos[1] - y0
        
        # 计算四个角点的影响
        def dot(v1, v2):
            return v1[...,0]*v2[0] + v1[...,1]*v2[1]
        
        # 梯度插值计算
        u = fx*fx*(3.0 - 2.0*fx)  # 缓和曲线
        v = fy*fy*(3.0 - 2.0*fy)
        
        # 双线性插值
        return (dot(gradients[x0,y0], [fx, fy]) * (1 - u) * (1 - v) + 
                dot(gradients[x1,y0], [fx-1, fy]) * u * (1 - v) + 
                dot(gradients[x0,y1], [fx, fy-1]) * (1 - u) * v + 
                dot(gradients[x1,y1], [fx-1, fy-1]) * u * v)
    
    # 使用向量化方法生成噪声
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 安全地计算所在位置
            pos_i = min(res[0] - 0.001, max(0, i/d[0]))
            pos_j = min(res[1] - 0.001, max(0, j/d[1]))
            noise[i,j] = perlin((pos_i, pos_j))
    
    # 归一化到[-1, 1]范围
    max_abs = np.max(np.abs(noise)) or 1.0  # 避免除以零
    return noise / max_abs
