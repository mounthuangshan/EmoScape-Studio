import os
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# 尝试导入地理数据处理库
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("警告: 未安装rasterio库，某些地理数据导入功能将不可用")

try:
    import elevation
    HAS_ELEVATION = True
except ImportError:
    HAS_ELEVATION = False
    print("警告: 未安装elevation库，SRTM数据下载功能将不可用")

class GeoDataImporter:
    """用于导入和处理真实地理数据的类"""
    
    def __init__(self, logger=None):
        """初始化地理数据导入器
        
        Args:
            logger: 日志记录器，用于输出进度和错误信息
        """
        self.logger = logger
        self.dem_data = None
        self.metadata = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查必要的依赖库是否已安装"""
        self.can_process_geotiff = HAS_RASTERIO
        self.can_download_srtm = HAS_ELEVATION
        
        if self.logger:
            if not self.can_process_geotiff:
                self.logger.log("警告: 未安装rasterio库，无法处理GeoTIFF文件", "WARNING")
            if not self.can_download_srtm:
                self.logger.log("警告: 未安装elevation库，无法下载SRTM数据", "WARNING")
    
    def log(self, message, level="INFO"):
        """记录日志信息"""
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def from_file(self, filepath):
        """从本地文件导入DEM数据
        
        Args:
            filepath: GeoTIFF或其他支持的栅格文件路径
            
        Returns:
            bool: 是否成功导入数据
        """
        if not self.can_process_geotiff:
            self.log("错误: 需要安装rasterio库才能从文件导入GeoTIFF", "ERROR")
            return False
            
        try:
            self.log(f"正在从文件导入地理数据: {filepath}")
            
            with rasterio.open(filepath) as src:
                self.dem_data = src.read(1)  # 读取第一个波段
                self.metadata = {
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "source": filepath
                }
                
            # 处理无效值
            if np.isnan(self.dem_data).any() or np.isinf(self.dem_data).any():
                self.log("检测到无效值，进行修复...")
                self.dem_data = np.nan_to_num(self.dem_data, nan=np.nanmean(self.dem_data))
            
            self.log(f"成功导入DEM数据, 大小: {self.dem_data.shape}")
            return True
            
        except Exception as e:
            self.log(f"导入地理数据失败: {str(e)}", "ERROR")
            return False
    
    def download_srtm(self, bounds, output_file=None):
        """下载指定边界的SRTM数据
        
        Args:
            bounds: (左, 下, 右, 上)坐标，表示经纬度边界
            output_file: 可选，保存下载数据的文件路径
            
        Returns:
            bool: 是否成功下载数据
        """
        if not self.can_download_srtm:
            self.log("错误: 需要安装elevation库才能下载SRTM数据", "ERROR")
            return False
            
        try:
            self.log(f"正在下载SRTM数据: 边界{bounds}")
            
            # 创建临时文件
            temp_file = output_file or "temp_srtm.tif"
            
            # 下载SRTM数据
            elevation.clip(bounds=bounds, output=temp_file)
            
            # 导入下载的数据
            result = self.from_file(temp_file)
            
            # 如果是临时文件且导入成功，则删除它
            if not output_file and os.path.exists(temp_file) and result:
                os.remove(temp_file)
                
            return result
            
        except Exception as e:
            self.log(f"下载SRTM数据失败: {str(e)}", "ERROR")
            return False
    
    def from_openstreetmap(self, location, width_km=10, height_km=10):
        """尝试从OpenStreetMap获取高度数据
        
        Args:
            location: 位置名称或(纬度,经度)元组
            width_km: 区域宽度，单位公里
            height_km: 区域高度，单位公里
            
        Returns:
            bool: 是否成功获取数据
        """
        try:
            # 这里是简化实现，实际需要调用地理编码API
            self.log(f"正在获取位置数据: {location}")
            self.log("注意: 此功能可能需要使用第三方API，具体实现略", "WARNING")
            return False
        except Exception as e:
            self.log(f"从OpenStreetMap获取数据失败: {str(e)}", "ERROR")
            return False
    
    def resize(self, target_width, target_height):
        """将DEM数据调整到目标大小
        
        Args:
            target_width: 目标宽度
            target_height: 目标高度
            
        Returns:
            numpy.ndarray: 调整大小后的DEM数据
        """
        if self.dem_data is None:
            self.log("错误: 没有可调整大小的DEM数据", "ERROR")
            return None
            
        try:
            # 简单的调整大小方法，可以根据需要选择更复杂的重采样方法
            import scipy.ndimage
            self.log(f"调整DEM数据大小: 从{self.dem_data.shape}到({target_height}, {target_width})")
            
            # 使用双线性插值进行缩放
            resized_data = scipy.ndimage.zoom(
                self.dem_data, 
                (target_height / self.dem_data.shape[0], target_width / self.dem_data.shape[1]),
                order=1
            )
            
            return resized_data
            
        except Exception as e:
            self.log(f"调整DEM数据大小失败: {str(e)}", "ERROR")
            return None
    
    def normalize(self, min_height=0.0, max_height=1.0):
        """将DEM数据归一化到指定范围
        
        Args:
            min_height: 目标最小高度值
            max_height: 目标最大高度值
            
        Returns:
            numpy.ndarray: 归一化后的DEM数据
        """
        if self.dem_data is None:
            self.log("错误: 没有可归一化的DEM数据", "ERROR")
            return None
            
        try:
            self.log(f"归一化DEM数据: 范围[{min_height}, {max_height}]")
            
            # 处理可能存在的异常值
            data = np.clip(self.dem_data, np.percentile(self.dem_data, 1), np.percentile(self.dem_data, 99))
            
            # 标准归一化
            data_min, data_max = data.min(), data.max()
            normalized = (data - data_min) / (data_max - data_min) * (max_height - min_height) + min_height
            
            return normalized
            
        except Exception as e:
            self.log(f"归一化DEM数据失败: {str(e)}", "ERROR")
            return None
    
    def get_height_map(self, target_width, target_height, normalize_range=(0.0, 1.0)):
        """获取处理好的高度图，用于游戏地图生成
        
        Args:
            target_width: 目标宽度
            target_height: 目标高度
            normalize_range: 归一化范围，默认为(0,1)
            
        Returns:
            numpy.ndarray: 处理好的高度图
        """
        if self.dem_data is None:
            self.log("错误: 没有可处理的DEM数据", "ERROR")
            return None
            
        try:
            # 调整大小
            resized = self.resize(target_width, target_height)
            if resized is None:
                return None
                
            # 归一化
            min_h, max_h = normalize_range
            normalized = self.normalize(min_h, max_h)
            if normalized is None:
                return None
                
            self.log(f"成功生成游戏高度图: 大小({target_height}, {target_width}), 高度范围[{min_h}, {max_h}]")
            return normalized
            
        except Exception as e:
            self.log(f"生成游戏高度图失败: {str(e)}", "ERROR")
            return None
    
    def visualize(self):
        """可视化当前加载的DEM数据"""
        if self.dem_data is None:
            self.log("错误: 没有可视化的DEM数据", "ERROR")
            return
            
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.dem_data, cmap='terrain')
            plt.colorbar(label='高度 (m)')
            plt.title('DEM数据可视化')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log(f"可视化DEM数据失败: {str(e)}", "ERROR")