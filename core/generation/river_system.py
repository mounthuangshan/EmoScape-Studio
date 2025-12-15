"""
河流生成系统 - 高性能水文学模拟
提供逼真的河流网络生成功能，基于地形水文学原理

特性:
- 多流向水流模拟
- 精确的流量累积计算
- 自然的河流蜿蜒度
- 基于物理的侵蚀模拟
- 高性能并行计算
"""

import numpy as np
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.ndimage import label, gaussian_filter, binary_dilation
from numba import jit, prange, float32, float64, int32, boolean, optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..services.map_tools import simplex_noise
#from ..services.map_tools import generate_noise_map

# 配置日志
logger = logging.getLogger(__name__)

# ==== 数据模型 ====

class FlowDirection(IntEnum):
    """流向枚举，使用D8方向编码"""
    NONE = 0
    E = 1
    SE = 2
    S = 4
    SW = 8
    W = 16
    NW = 32
    N = 64
    NE = 128


@dataclass
class RiverGenerationConfig:
    """河流生成配置"""
    # 基本参数
    min_watershed_size: int = 50            # 最小流域面积
    precipitation_factor: float = 1.0       # 降水系数
    meander_factor: float = 0.3             # 河流弯曲系数
    seed: Optional[int] = None              # 随机种子
    
    # 高级参数
    flow_exponent: float = 10.0             # 流量分配指数
    smoothing_sigma: float = 1.0            # 地形平滑系数
    erosion_strength: float = 0.3           # 侵蚀强度
    random_factor_range: Tuple[float, float] = (0.8, 1.2)  # 随机因子范围
    
    # 性能参数
    use_parallel: bool = True               # 使用并行计算
    chunk_size: Optional[int] = None        # 分块大小，None为不分块
    precision: str = 'float32'              # 计算精度: 'float32'或'float64'
    
    # 处理参数
    progress_callback: Optional[Callable[[str, float], None]] = None  # 进度回调
    verbose: bool = True                    # 是否打印详细信息


class FlowModel:
    """水流模型 - 负责计算流向和流量"""
    
    def __init__(self, config: RiverGenerationConfig):
        """初始化水流模型
        
        Args:
            config: 河流生成配置
        """
        self.config = config
        self._dtype = np.float32 if config.precision == 'float32' else np.float64
        
        # D8方向与编码
        self.directions = [
            (0, 1),     # E  - 1
            (1, 1),     # SE - 2
            (1, 0),     # S  - 4
            (1, -1),    # SW - 8
            (0, -1),    # W  - 16
            (-1, -1),   # NW - 32
            (-1, 0),    # N  - 64
            (-1, 1)     # NE - 128
        ]
        self.dir_codes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.distances = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=self._dtype)
        
        # 设置随机种子
        self.seed = 42 if config.seed is None else config.seed
        np.random.seed(self.seed)
    
    def compute_flow_directions(self, height_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算D8流向场
        
        Args:
            height_map: 高度图数组
            
        Returns:
            主流向数组和多流向权重数组
        """
        height, width = height_map.shape
        
        # 创建平滑版地形
        if self.config.verbose:
            logger.info("平滑地形用于流向计算...")
        
        smoothed_height = gaussian_filter(height_map, sigma=self.config.smoothing_sigma)
        
        # 初始化流向数组
        flow_dirs = np.zeros((height, width, 8), dtype=self._dtype)
        flow_dir_main = np.zeros((height, width), dtype=np.int32)
        
        if self.config.verbose:
            logger.info("计算D8多流向场...")
            
        if self.config.use_parallel:
            # 并行版本
            flow_dir_main, flow_dirs = self._compute_flow_directions_parallel(
                smoothed_height, height, width)
        else:
            # 非并行版本
            flow_dir_main, flow_dirs = self._compute_flow_directions_sequential(
                smoothed_height, height, width)
            
        return flow_dir_main, flow_dirs
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_flow_directions_parallel(
        smoothed_height: np.ndarray, 
        height: int, 
        width: int,
        meander_factor: float = 0.3,
        flow_exponent: float = 10.0,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """并行计算D8流向场 (Numba加速)"""
        # 初始化输出数组
        flow_dirs = np.zeros((height, width, 8), dtype=np.float32)
        flow_dir_main = np.zeros((height, width), dtype=np.int32)
        
        # 8个邻居方向和对应编码
        neighbors = np.array([
            [0, 1],     # E  - 1
            [1, 1],     # SE - 2
            [1, 0],     # S  - 4
            [1, -1],    # SW - 8
            [0, -1],    # W  - 16
            [-1, -1],   # NW - 32
            [-1, 0],    # N  - 64
            [-1, 1]     # NE - 128
        ], dtype=np.int32)
        dir_codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
        distances = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)
        
        # 并行处理每个单元格
        for y in prange(1, height-1):
            for x in range(1, width-1):
                center_height = smoothed_height[y, x]
                
                # 计算所有方向的坡度
                slopes = np.zeros(8, dtype=np.float32)
                max_slope = 0.0
                max_dir_idx = -1
                flat_area = True
                
                for i in range(8):
                    dy, dx = neighbors[i]
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < height and 0 <= nx < width:
                        # 添加蜿蜒度的随机因子
                        random_factor = 1.0
                        
                        # 简化的哈希噪声代替simplex_noise (Numba兼容)
                        h = ((x * 12.9898 + y * 78.233 + i * 37.719 + seed) % 100) / 100.0
                        noise_val = 2.0 * h - 1.0
                        
                        if flat_area or max_slope < 0.001:
                            random_factor = 1.0 + meander_factor * noise_val * 0.8
                        elif meander_factor > 0:
                            random_factor = 1.0 + meander_factor * noise_val * 0.2
                        
                        # 计算坡度
                        slope = (center_height - smoothed_height[ny, nx]) / distances[i] * random_factor
                        
                        # 平地流向偏好
                        if abs(slope) < 0.001:
                            h2 = ((x * 4.898 + y * 7.233 + seed + 100 + i) % 100) / 100.0
                            large_scale_pref = (2.0 * h2 - 1.0) * 0.002
                            slope += large_scale_pref
                        
                        if slope > 0.001:
                            flat_area = False
                        
                        slopes[i] = max(slope, 0.0)  # 只保留正值（下坡方向）
                        
                        if slope > max_slope:
                            max_slope = slope
                            max_dir_idx = i
                
                # 设置主流向
                if max_dir_idx >= 0:
                    flow_dir_main[y, x] = dir_codes[max_dir_idx]
                
                # 多流向分配
                if flat_area or max_slope <= 0:
                    # 平地上使用微弱的随机偏好
                    for i in range(8):
                        h3 = ((x + i) * 10.9898 + (y + i) * 8.233 + seed + i) % 100 / 100.0
                        rand_bias = (2.0 * h3 - 1.0) * 0.01
                        flow_dirs[y, x, i] = 0.1 + rand_bias
                else:
                    # 计算流量权重
                    total_weight = 0.0
                    
                    for i in range(8):
                        if slopes[i] > 0:
                            # 使用坡度的幂律分配
                            weight = (slopes[i] / max_slope) ** flow_exponent
                            flow_dirs[y, x, i] = weight
                            total_weight += weight
                    
                    # 归一化权重
                    if total_weight > 0:
                        for i in range(8):
                            flow_dirs[y, x, i] /= total_weight
        
        return flow_dir_main, flow_dirs
    
    def _compute_flow_directions_sequential(
        self, smoothed_height: np.ndarray, height: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """串行计算D8流向场"""
        # 初始化输出数组
        flow_dirs = np.zeros((height, width, 8), dtype=self._dtype)
        flow_dir_main = np.zeros((height, width), dtype=np.int32)
        
        # 处理进度
        total_cells = (height-2) * (width-2)
        processed = 0
        
        for y in range(1, height-1):
            for x in range(1, width-1):
                center_height = smoothed_height[y, x]
                
                # 计算所有下降方向及其坡度
                slopes = []
                max_slope = 0
                max_dir_idx = -1
                flat_area = True
                
                for i, (dy, dx) in enumerate(self.directions):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        # 计算坡度（添加蜿蜒度因子）
                        random_factor = 1.0
                        if flat_area or max_slope < 0.001:
                            noise_val = simplex_noise((x+i)/width*70, (y+i)/height*70, self.seed + i*13)
                            random_factor = 1.0 + self.config.meander_factor * noise_val * 0.8
                        elif self.config.meander_factor > 0:
                            noise_val = simplex_noise(x/width*50, y/height*50, self.seed + i)
                            random_factor = 1.0 + self.config.meander_factor * noise_val * 0.2
                        
                        slope = (center_height - smoothed_height[ny, nx]) / self.distances[i] * random_factor
                        
                        # 平地流向偏好
                        if abs(slope) < 0.001:
                            large_scale_pref = simplex_noise(x/width*5, y/height*5, self.seed+100+i) * 0.002
                            slope += large_scale_pref
                        
                        if slope > 0.001:
                            flat_area = False
                        
                        slopes.append((slope, i))
                        
                        if slope > max_slope:
                            max_slope = slope
                            max_dir_idx = i
                
                # 设置主流向
                if max_dir_idx >= 0:
                    flow_dir_main[y, x] = self.dir_codes[max_dir_idx]
                
                # 多流向分配
                if flat_area or max_slope <= 0:
                    for i in range(8):
                        rand_bias = simplex_noise((x+i)/width*100, (y+i)/height*100, self.seed+i) * 0.01
                        flow_dirs[y, x, i] = 0.1 + rand_bias
                else:
                    total_weight = 0
                    
                    for slope, i in slopes:
                        if slope > 0:
                            weight = (slope / max_slope) ** self.config.flow_exponent
                            flow_dirs[y, x, i] = weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        for i in range(8):
                            flow_dirs[y, x, i] /= total_weight
                
                # 更新进度
                processed += 1
                if self.config.progress_callback and processed % 1000 == 0:
                    progress = processed / total_cells
                    self.config.progress_callback("计算流向", progress)
        
        return flow_dir_main, flow_dirs
    
    def compute_flow_accumulation(
        self, height_map: np.ndarray, flow_dir: np.ndarray
    ) -> np.ndarray:
        """计算流量累积
        
        Args:
            height_map: 高度图数组
            flow_dir: 主流向数组
            
        Returns:
            流量累积数组
        """
        height, width = height_map.shape
        
        if self.config.verbose:
            logger.info("计算降雨分布...")
        
        # 计算降雨分布
        rainfall_map = self._compute_rainfall(height_map)
        
        if self.config.verbose:
            logger.info("计算流量累积...")
        
        # 计算流量累积
        flow_accum = np.zeros((height, width), dtype=self._dtype)
        contribution_map = rainfall_map.copy()
        
        # 从高到低进行流量累积
        cell_heights = []
        for y in range(height):
            for x in range(width):
                cell_heights.append((height_map[y, x], y, x))
        
        cell_heights.sort(reverse=True)
        total_cells = len(cell_heights)
        
        # 进度跟踪
        if self.config.verbose:
            cell_iterator = tqdm(cell_heights, desc="累积流量", unit="单元格")
        else:
            cell_iterator = cell_heights
            
        for i, (_, y, x) in enumerate(cell_iterator):
            flow_accum[y, x] += contribution_map[y, x]
            
            # 获取流向
            dir_code = flow_dir[y, x]
            if dir_code == 0:  # 无流向
                continue
            
            # 计算下一个单元格
            next_y, next_x = self._get_next_cell(y, x, dir_code)
            
            # 检查是否在范围内
            if 0 <= next_y < height and 0 <= next_x < width:
                # 将当前单元格的贡献传递给下游单元格
                contribution_map[next_y, next_x] += flow_accum[y, x]
            
            # 更新进度
            if self.config.progress_callback and i % 1000 == 0:
                self.config.progress_callback("累积流量", i / total_cells)
        
        return flow_accum
    
    def _compute_rainfall(self, height_map: np.ndarray) -> np.ndarray:
        """计算降雨分布
        
        Args:
            height_map: 高度图数组
            
        Returns:
            降雨量数组
        """
        height, width = height_map.shape
        rainfall_map = np.ones((height, width), dtype=self._dtype)
        
        # 使用向量化操作优化降雨分布计算
        if self.config.use_parallel:
            rainfall_map = self._compute_rainfall_parallel(
                height_map, rainfall_map, self.config.precipitation_factor, 
                self.config.random_factor_range, self.seed)
        else:
            # 非并行版本
            min_factor, max_factor = self.config.random_factor_range
            factor_range = max_factor - min_factor
            
            for y in range(height):
                for x in range(width):
                    # 随机变化
                    random_factor = min_factor + factor_range * np.random.random()
                    
                    # 高度影响降雨
                    elevation_factor = 1.0
                    if height_map[y, x] > 60:  # 高山地区
                        elevation_factor = 1.2
                    
                    # 山脊检测
                    is_local_peak = True
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < height and 0 <= nx < width and 
                                height_map[ny, nx] > height_map[y, x]):
                                is_local_peak = False
                                break
                        if not is_local_peak:
                            break
                    
                    peak_factor = 1.2 if is_local_peak else 1.0
                    rainfall_map[y, x] = self.config.precipitation_factor * random_factor * elevation_factor * peak_factor
        
        return rainfall_map
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_rainfall_parallel(
        height_map: np.ndarray, 
        rainfall_map: np.ndarray,
        precipitation_factor: float,
        random_factor_range: Tuple[float, float],
        seed: int
    ) -> np.ndarray:
        """并行计算降雨分布 (Numba加速)"""
        height, width = height_map.shape
        min_factor, max_factor = random_factor_range
        factor_range = max_factor - min_factor
        
        # 为随机数生成设置种子
        np.random.seed(seed)
        
        for y in prange(height):
            for x in range(width):
                # 使用简单的伪随机数生成
                rand_val = ((x * 15.487) + (y * 35.731) + seed) % 997 / 997.0
                random_factor = min_factor + factor_range * rand_val
                
                # 高度影响降雨
                elevation_factor = 1.0
                if height_map[y, x] > 60:  # 高山地区
                    elevation_factor = 1.2
                
                # 简化的山脊检测 - 检查局部最大值
                is_local_peak = True
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            height_map[ny, nx] > height_map[y, x]):
                            is_local_peak = False
                            break
                    if not is_local_peak:
                        break
                
                peak_factor = 1.2 if is_local_peak else 1.0
                rainfall_map[y, x] = precipitation_factor * random_factor * elevation_factor * peak_factor
        
        return rainfall_map
    
    def _get_next_cell(self, y: int, x: int, dir_code: int) -> Tuple[int, int]:
        """根据流向获取下一个单元格坐标"""
        next_y, next_x = y, x
        
        if dir_code & 1:    # E
            next_x += 1
        elif dir_code & 2:   # SE
            next_y += 1
            next_x += 1
        elif dir_code & 4:   # S
            next_y += 1
        elif dir_code & 8:   # SW
            next_y += 1
            next_x -= 1
        elif dir_code & 16:  # W
            next_x -= 1
        elif dir_code & 32:  # NW
            next_y -= 1
            next_x -= 1
        elif dir_code & 64:  # N
            next_y -= 1
        elif dir_code & 128: # NE
            next_y -= 1
            next_x += 1
        
        return next_y, next_x


class RiverExtractor:
    """河流提取器 - 负责从流量数据提取河流特征"""
    
    def __init__(self, config: RiverGenerationConfig, flow_model: FlowModel):
        """初始化河流提取器
        
        Args:
            config: 河流生成配置
            flow_model: 流量模型
        """
        self.config = config
        self.flow_model = flow_model
        self._dtype = np.float32 if config.precision == 'float32' else np.float64
    
    def extract_rivers(
        self, height_map: np.ndarray, flow_dir: np.ndarray, flow_accum: np.ndarray
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """提取河流网络
        
        Args:
            height_map: 高度图数组
            flow_dir: 主流向数组
            flow_accum: 流量累积数组
            
        Returns:
            河流掩码和河流路径列表
        """
        height, width = height_map.shape
        result = height_map.copy()
        
        if self.config.verbose:
            logger.info("生成河流网络...")
        
        # 标记河流单元格
        river_mask = flow_accum > self.config.min_watershed_size
        potential_river = flow_accum > (self.config.min_watershed_size * 0.7)
        
        # 使用形态学操作连接潜在河流
        potential_river = binary_dilation(potential_river, iterations=1)
        
        if self.config.verbose:
            logger.info("应用河流侵蚀...")
        
        # 应用河流侵蚀
        result = self._apply_river_erosion(result, river_mask, flow_dir, flow_accum)
        
        if self.config.verbose:
            logger.info("提取河流路径...")
        
        # 提取河流路径
        river_features = self._extract_river_paths(result, river_mask, flow_dir)
        
        return river_mask, river_features
    
    def _apply_river_erosion(
        self, height_map: np.ndarray, river_mask: np.ndarray, 
        flow_dir: np.ndarray, flow_accum: np.ndarray
    ) -> np.ndarray:
        """应用河流侵蚀
        
        Args:
            height_map: 高度图数组
            river_mask: 河流掩码
            flow_dir: 主流向数组
            flow_accum: 流量累积数组
            
        Returns:
            侵蚀后的高度图
        """
        height, width = height_map.shape
        result = height_map.copy()
        
        # 如果启用并行，使用并行版本的侵蚀
        if self.config.use_parallel:
            result = self._apply_river_erosion_parallel(
                result, river_mask, flow_dir, flow_accum, 
                self.config.erosion_strength)
        else:
            # 按行列遍历应用侵蚀
            total_erosions = np.sum(river_mask)
            erosions_done = 0
            
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if river_mask[y, x]:
                        # 河流侵蚀量与流量和坡度成正比
                        flow_volume = min(flow_accum[y, x] / 1000, 10.0)
                        
                        # 计算局部坡度
                        local_slope = 0.0
                        if flow_dir[y, x] != 0:
                            # 获取下一个单元格
                            next_y, next_x = self.flow_model._get_next_cell(y, x, flow_dir[y, x])
                            
                            if 0 <= next_y < height and 0 <= next_x < width:
                                height_diff = result[y, x] - result[next_y, next_x]
                                # 计算距离 (直线或对角线)
                                dir_code = flow_dir[y, x]
                                distance = 1.0 if (dir_code & (1|4|16|64)) else 1.414
                                local_slope = height_diff / distance
                        
                        # 根据河流大小决定侵蚀参数
                        if flow_volume > 5.0:  # 大河流
                            erosion_width = 3
                            erosion_factor = self.config.erosion_strength
                        else:  # 小河流
                            erosion_width = 1
                            erosion_factor = self.config.erosion_strength * 0.7
                        
                        # 应用侵蚀
                        for wy in range(max(0, y-erosion_width), min(height, y+erosion_width+1)):
                            for wx in range(max(0, x-erosion_width), min(width, x+erosion_width+1)):
                                # 计算到河流中心的距离
                                dist = np.sqrt((wy-y)**2 + (wx-x)**2)
                                
                                if dist <= erosion_width:
                                    # 基于距离计算侵蚀量
                                    dist_factor = 1.0 - dist/erosion_width
                                    erosion_amount = (erosion_factor * flow_volume * 
                                                    (0.05 + local_slope * 0.5) * 
                                                    dist_factor)
                                    
                                    # 设置最大侵蚀深度限制
                                    max_erosion = 0.2 + 0.3 * flow_volume
                                    erosion_amount = min(erosion_amount, max_erosion)
                                    
                                    # 应用侵蚀
                                    result[wy, wx] -= erosion_amount
                                    
                                    # 河道宽度随流量变化
                                    if dist < 1.0:
                                        width_factor = 0.5 + 0.5 * min(1.0, flow_volume / 3.0)
                                        edge_smooth = 0.5 - 0.5 * np.cos(np.pi * dist / width_factor)
                                        result[wy, wx] -= erosion_amount * 0.2 * edge_smooth
                        
                        # 更新进度
                        erosions_done += 1
                        if self.config.progress_callback and erosions_done % 100 == 0:
                            self.config.progress_callback("河流侵蚀", erosions_done / total_erosions)
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _apply_river_erosion_parallel(
        height_map: np.ndarray,
        river_mask: np.ndarray,
        flow_dir: np.ndarray,
        flow_accum: np.ndarray,
        erosion_strength: float
    ) -> np.ndarray:
        """并行应用河流侵蚀 (Numba加速)"""
        height, width = height_map.shape
        result = height_map.copy()
        
        for y in prange(1, height-1):
            for x in range(1, width-1):
                if river_mask[y, x]:
                    # 河流侵蚀量与流量和坡度成正比
                    flow_volume = min(flow_accum[y, x] / 1000, 10.0)
                    
                    # 计算局部坡度
                    local_slope = 0.0
                    if flow_dir[y, x] != 0:
                        # 查找下游单元格
                        dir_code = flow_dir[y, x]
                        next_y, next_x = y, x
                        
                        if dir_code & 1:    # E
                            next_x += 1
                        elif dir_code & 2:   # SE
                            next_y += 1
                            next_x += 1
                        elif dir_code & 4:   # S
                            next_y += 1
                        elif dir_code & 8:   # SW
                            next_y += 1
                            next_x -= 1
                        elif dir_code & 16:  # W
                            next_x -= 1
                        elif dir_code & 32:  # NW
                            next_y -= 1
                            next_x -= 1
                        elif dir_code & 64:  # N
                            next_y -= 1
                        elif dir_code & 128: # NE
                            next_y -= 1
                            next_x += 1
                        
                        if 0 <= next_y < height and 0 <= next_x < width:
                            height_diff = result[y, x] - result[next_y, next_x]
                            # 计算距离 (直线或对角线)
                            distance = 1.0 if (dir_code & (1|4|16|64)) else 1.414
                            local_slope = height_diff / distance
                    
                    # 根据河流大小决定侵蚀参数
                    if flow_volume > 5.0:  # 大河流
                        erosion_width = 3
                        erosion_factor = erosion_strength
                    else:  # 小河流
                        erosion_width = 1
                        erosion_factor = erosion_strength * 0.7
                    
                    # 应用侵蚀
                    for wy in range(max(0, y-erosion_width), min(height, y+erosion_width+1)):
                        for wx in range(max(0, x-erosion_width), min(width, x+erosion_width+1)):
                            # 计算到河流中心的距离
                            dist = np.sqrt((wy-y)**2 + (wx-x)**2)
                            
                            if dist <= erosion_width:
                                # 基于距离计算侵蚀量
                                dist_factor = 1.0 - dist/erosion_width
                                erosion_amount = (erosion_factor * flow_volume * 
                                                (0.05 + local_slope * 0.5) * 
                                                dist_factor)
                                
                                # 设置最大侵蚀深度限制
                                max_erosion = 0.2 + 0.3 * flow_volume
                                erosion_amount = min(erosion_amount, max_erosion)
                                
                                # 应用侵蚀
                                result[wy, wx] -= erosion_amount
                                
                                # 河道宽度随流量变化
                                if dist < 1.0:
                                    width_factor = 0.5 + 0.5 * min(1.0, flow_volume / 3.0)
                                    edge_smooth = 0.5 - 0.5 * np.cos(np.pi * dist / width_factor)
                                    result[wy, wx] -= erosion_amount * 0.2 * edge_smooth
        
        return result
    
    def _extract_river_paths(
        self, height_map: np.ndarray, river_mask: np.ndarray, flow_dir: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """提取河流路径
        
        Args:
            height_map: 高度图数组
            river_mask: 河流掩码
            flow_dir: 主流向数组
            
        Returns:
            河流路径列表
        """
        height, width = height_map.shape
        
        # 标记河流连通区域
        river_labels, num_features = label(river_mask)
        river_features = []
        
        if self.config.verbose:
            feature_iterator = tqdm(range(1, num_features + 1), desc="处理河流", unit="条")
        else:
            feature_iterator = range(1, num_features + 1)
        
        for i in feature_iterator:
            # 提取当前河流的坐标
            river_coords = np.argwhere(river_labels == i)
            
            if len(river_coords) > 5:  # 忽略太短的河流
                # 查找河流的源头
                start_y, start_x = self._find_river_source(river_coords, height_map, flow_dir)
                
                # 追踪河流路径
                river_path = self._trace_river_path(
                    start_y, start_x, river_mask, flow_dir, height, width)
                
                if len(river_path) > 5:  # 保存足够长的河流
                    river_features.append(river_path)
                    
                    # 应用河流蜿蜒度增强
                    if self.config.meander_factor > 0 and len(river_path) > 10:
                        smoothed_path = self._smooth_river_path(river_path, height, width)
                        
                        # 使用平滑路径替换原始路径
                        river_features[-1] = smoothed_path
        
        return river_features
    
    def _find_river_source(
        self, river_coords: np.ndarray, height_map: np.ndarray, flow_dir: np.ndarray
    ) -> Tuple[int, int]:
        """查找河流源头
        
        Args:
            river_coords: 河流坐标数组
            height_map: 高度图数组
            flow_dir: 主流向数组
            
        Returns:
            源头坐标 (y, x)
        """
        # 计算入流数量
        inflow_counts = np.zeros(len(river_coords), dtype=np.int32)
        
        for idx, (cy, cx) in enumerate(river_coords):
            # 检查有多少河流点流向该点
            for y, x in river_coords:
                if (y, x) != (cy, cx):
                    dir_code = flow_dir[y, x]
                    next_y, next_x = self.flow_model._get_next_cell(y, x, dir_code)
                    
                    if next_y == cy and next_x == cx:
                        inflow_counts[idx] += 1
        
        # 源头应该是没有或很少有入流的点
        potential_sources = []
        for idx, (y, x) in enumerate(river_coords):
            if inflow_counts[idx] <= 1:
                potential_sources.append((y, x, height_map[y, x]))
        
        # 如果找到了可能的源头，选择高度最高的作为实际源头
        if potential_sources:
            potential_sources.sort(key=lambda p: p[2], reverse=True)
            start_y, start_x, _ = potential_sources[0]
        else:
            # 如果没有明确的源头，选择最高的点
            elevations = [height_map[y, x] for y, x in river_coords]
            start_idx = np.argmax(elevations)
            start_y, start_x = river_coords[start_idx]
        
        return start_y, start_x
    
    def _trace_river_path(
        self, start_y: int, start_x: int, river_mask: np.ndarray, 
        flow_dir: np.ndarray, height: int, width: int
    ) -> List[Tuple[int, int]]:
        """追踪河流路径
        
        Args:
            start_y, start_x: 起始坐标
            river_mask: 河流掩码
            flow_dir: 主流向数组
            height, width: 地图尺寸
            
        Returns:
            河流路径
        """
        river_path = [(start_y, start_x)]
        curr_y, curr_x = start_y, start_x
        visited = set([(start_y, start_x)])
        
        while True:
            if flow_dir[curr_y, curr_x] == 0:
                break
                
            # 获取下一个位置
            next_y, next_x = self.flow_model._get_next_cell(curr_y, curr_x, flow_dir[curr_y, curr_x])
            
            # 检查是否有效且在河流上
            if (0 <= next_y < height and 0 <= next_x < width and 
                river_mask[next_y, next_x] and 
                (next_y, next_x) not in visited):
                
                river_path.append((next_y, next_x))
                visited.add((next_y, next_x))
                curr_y, curr_x = next_y, next_x
            else:
                break
            
            # 防止无限循环
            if len(river_path) > min(height, width) * 2:
                break
        
        return river_path
    
    def _smooth_river_path(
        self, river_path: List[Tuple[int, int]], height: int, width: int
    ) -> List[Tuple[int, int]]:
        """平滑河流路径
        
        Args:
            river_path: 原始河流路径
            height, width: 地图尺寸
            
        Returns:
            平滑后的河流路径
        """
        smoothed_path = []
        
        # 应用移动平均平滑
        window_size = 3
        for i in range(len(river_path)):
            # 计算窗口内点的平均位置
            window_start = max(0, i - window_size // 2)
            window_end = min(len(river_path), i + window_size // 2 + 1)
            window_points = river_path[window_start:window_end]
            
            avg_y = sum(p[0] for p in window_points) / len(window_points)
            avg_x = sum(p[1] for p in window_points) / len(window_points)
            
            # 应用平滑但保持一定的原始变化
            smooth_factor = 0.7  # 平滑系数
            new_y = int(river_path[i][0] * (1-smooth_factor) + avg_y * smooth_factor)
            new_x = int(river_path[i][1] * (1-smooth_factor) + avg_x * smooth_factor)
            
            # 确保坐标在有效范围内
            new_y = max(0, min(height-1, new_y))
            new_x = max(0, min(width-1, new_x))
            
            smoothed_path.append((new_y, new_x))
        
        return smoothed_path


class RiverGenerator:
    """河流生成器 - 主类，协调整个河流生成过程"""
    
    def __init__(self, config: Optional[RiverGenerationConfig] = None):
        """初始化河流生成器
        
        Args:
            config: 河流生成配置，如果为None则使用默认配置
        """
        self.config = config or RiverGenerationConfig()
        self.flow_model = FlowModel(self.config)
        self.river_extractor = RiverExtractor(self.config, self.flow_model)
    
    def generate(self, height_map: np.ndarray) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], np.ndarray]:
        """生成河流
        
        Args:
            height_map: 高度图数组
            
        Returns:
            元组: (修改后的高度图, 河流路径列表, 流量累积图)
        """
        if self.config.verbose:
            logger.info("开始生成河流...")
        
        height, width = height_map.shape
        
        # 计算流向
        flow_dir, flow_dirs = self.flow_model.compute_flow_directions(height_map)
        
        # 计算流量累积
        flow_accum = self.flow_model.compute_flow_accumulation(height_map, flow_dir)
        
        # 提取河流
        river_mask, river_features = self.river_extractor.extract_rivers(
            height_map, flow_dir, flow_accum)
        
        # 获取修改后的高度图
        updated_height_map = height_map.copy()
        for y in range(height):
            for x in range(width):
                if river_mask[y, x]:
                    # 河流路径略微降低地形
                    updated_height_map[y, x] = height_map[y, x] - 0.5
        
        if self.config.verbose:
            logger.info(f"河流生成完成，共生成 {len(river_features)} 条河流")
        
        return updated_height_map, river_features, flow_accum
    
    def visualize(
        self, height_map: np.ndarray, river_features: List[List[Tuple[int, int]]], 
        flow_accum: Optional[np.ndarray] = None, show: bool = True, 
        save_path: Optional[str] = None
    ) -> None:
        """可视化河流生成结果
        
        Args:
            height_map: 高度图数组
            river_features: 河流路径列表
            flow_accum: 流量累积数组
            show: 是否显示图像
            save_path: 保存路径
        """
        height, width = height_map.shape
        
        # 创建河流掩码
        river_mask = np.zeros((height, width), dtype=bool)
        for river_path in river_features:
            for y, x in river_path:
                if 0 <= y < height and 0 <= x < width:
                    river_mask[y, x] = True
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3 if flow_accum is not None else 2, figsize=(18, 6))
        
        # 地形图
        terrain_img = axes[0].imshow(height_map, cmap='terrain')
        axes[0].set_title('地形')
        fig.colorbar(terrain_img, ax=axes[0], shrink=0.6)
        
        # 河流图
        terrain_with_rivers = height_map.copy()
        terrain_with_rivers[river_mask] = np.min(height_map) - 5  # 使河流更明显
        river_img = axes[1].imshow(terrain_with_rivers, cmap='terrain')
        axes[1].set_title('河流网络')
        fig.colorbar(river_img, ax=axes[1], shrink=0.6)
        
        # 流量累积图
        if flow_accum is not None:
            # 对数变换以突出小河流
            log_flow = np.log1p(flow_accum)
            flow_img = axes[2].imshow(log_flow, cmap='Blues')
            axes[2].set_title('流量累积 (对数)')
            fig.colorbar(flow_img, ax=axes[2], shrink=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()