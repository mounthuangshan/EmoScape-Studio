import numpy as np
from PIL import Image
import colorsys
import random

class StyleTransformer:
    """地图样式转换基类"""
    
    def __init__(self, map_data, config=None):
        self.map_data = map_data
        self.config = config or {}
        
    def transform(self):
        """转换地图数据并返回结果"""
        raise NotImplementedError("子类必须实现transform()")
        
    def get_preview_image(self, width=512, height=512):
        """生成转换后的预览图像"""
        raise NotImplementedError("子类必须实现get_preview_image()")


class MinecraftStyleTransformer(StyleTransformer):
    """将地图数据转换为Minecraft风格"""
    
    # Minecraft生物群系颜色映射
    BIOME_COLORS = {
        0: (0, 124, 15),    # 平原（草地）
        1: (250, 148, 24),  # 沙漠（沙地）
        2: (5, 102, 33),    # 森林（深色草地）
        3: (96, 96, 96),    # 山脉（石头）
        4: (28, 163, 236),  # 海洋（水）
        5: (19, 119, 252),  # 河流（水）
        6: (238, 238, 238), # 雪地（雪）
        7: (71, 193, 97),   # 丛林（丛林草地）
        8: (150, 150, 150), # 石滩（沙砾）
        9: (218, 210, 158), # 海滩（沙子）
        10: (73, 129, 60),  # 沼泽（沼泽草地）
        11: (136, 36, 36),  # 下界（下界岩）
        12: (240, 175, 21), # 热带草原（热带草原草地）
        13: (25, 74, 35),   # 针叶林（针叶林草地）
        14: (120, 152, 120),# 峭壁（峭壁草地）
        15: (240, 240, 240),# 冰原（雪）
        # 未知生物群系的默认值
        "default": (0, 124, 15) # 默认为平原
    }
    
    def __init__(self, map_data, config=None):
        super().__init__(map_data, config)
        # Minecraft特定设置
        self.max_height = self.config.get("max_height", 255)
        self.sea_level = self.config.get("sea_level", 63)
        self.block_resolution = self.config.get("block_resolution", 1)  # 每个地图单元格对应1个方块
        
    def transform(self):
        """将地图数据转换为Minecraft风格"""
        # 创建一个新的字典来存储转换后的数据，而不是尝试复制MapData对象
        minecraft_map_data = {
            "layers": {},
            "params": self.map_data.params.copy() if hasattr(self.map_data, "params") else {},
            "width": self.map_data.width if hasattr(self.map_data, "width") else 0,
            "height": self.map_data.height if hasattr(self.map_data, "height") else 0
        }
        
        # 复制所有层
        if hasattr(self.map_data, "layers"):
            for key, layer in self.map_data.layers.items():
                if isinstance(layer, np.ndarray):
                    minecraft_map_data["layers"][key] = layer.copy()
                elif isinstance(layer, list):
                    minecraft_map_data["layers"][key] = [item.copy() if hasattr(item, "copy") else dict(item) for item in layer]
                else:
                    minecraft_map_data["layers"][key] = layer
        
        # 转换高度图到Minecraft高度范围（0-255）
        if "height" in minecraft_map_data["layers"]:
            height_map = minecraft_map_data["layers"]["height"]
            if height_map is not None:
                # 标准化高度到0-1范围
                if np.max(height_map) > 1.0:
                    height_map = height_map / np.max(height_map)
                
                # 缩放到Minecraft高度范围，保持海平面在self.sea_level
                minecraft_height = np.zeros_like(height_map)
                water_threshold = self.config.get("water_threshold", 0.4)
                underwater_mask = height_map < water_threshold
                land_mask = ~underwater_mask
                
                # 水下地形：0到sea_level-5
                minecraft_height[underwater_mask] = self.sea_level - 5 + (height_map[underwater_mask] / water_threshold) * 5
                
                # 陆地：sea_level到max_height
                height_range = self.max_height - self.sea_level
                minecraft_height[land_mask] = self.sea_level + ((height_map[land_mask] - water_threshold) / (1 - water_threshold)) * height_range
                
                minecraft_height = np.round(minecraft_height).astype(np.int32)
                minecraft_map_data["layers"]["height"] = minecraft_height
                
                # 添加Minecraft水层
                water_layer = np.zeros_like(height_map, dtype=np.bool_)
                water_layer[height_map < water_threshold] = True
                minecraft_map_data["layers"]["minecraft_water"] = water_layer
        
        # 转换植被到Minecraft方块类型
        if "vegetation" in minecraft_map_data["layers"]:
            # 使用get_layer函数访问植被
            vegetation = self.map_data.get_layer("vegetation")
            if vegetation and isinstance(vegetation, list):
                minecraft_vegetation = []
                for veg in vegetation:
                    mc_veg = dict(veg)
                    veg_type = veg.get("type", "tree")
                    if veg_type in ["tree", "oak"]:
                        mc_veg["minecraft_block"] = "oak_log"
                    elif veg_type in ["pine", "spruce"]:
                        mc_veg["minecraft_block"] = "spruce_log"
                    elif veg_type in ["birch"]:
                        mc_veg["minecraft_block"] = "birch_log"
                    elif veg_type in ["jungle"]:
                        mc_veg["minecraft_block"] = "jungle_log"
                    elif veg_type in ["acacia"]:
                        mc_veg["minecraft_block"] = "acacia_log"
                    elif veg_type in ["bush", "shrub"]:
                        mc_veg["minecraft_block"] = "azalea_bush"
                    elif veg_type in ["cactus"]:
                        mc_veg["minecraft_block"] = "cactus"
                    elif veg_type in ["flower"]:
                        mc_veg["minecraft_block"] = random.choice([
                            "dandelion", "poppy", "blue_orchid", "allium", 
                            "azure_bluet", "red_tulip", "orange_tulip", 
                            "white_tulip", "pink_tulip", "oxeye_daisy"
                        ])
                    else:
                        mc_veg["minecraft_block"] = "oak_log"  # 默认值
                    
                    minecraft_vegetation.append(mc_veg)
                
                minecraft_map_data["layers"]["vegetation"] = minecraft_vegetation
        
        # 转换建筑到Minecraft结构
        if "buildings" in minecraft_map_data["layers"]:
            # 使用get_layer函数访问建筑
            buildings = self.map_data.get_layer("buildings")
            if buildings and isinstance(buildings, list):
                minecraft_buildings = []
                for building in buildings:
                    mc_building = dict(building)
                    building_type = building.get("type", "house")
                    if building_type in ["house", "small_house"]:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_small_house_1"
                    elif building_type in ["large_house", "mansion"]:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_big_house_1"
                    elif building_type in ["shop", "store"]:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_butcher_shop_1"
                    elif building_type in ["temple", "church"]:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_temple_1"
                    elif building_type in ["farm"]:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_farm_1"
                    elif building_type in ["castle", "fortress"]:
                        mc_building["minecraft_structure"] = "pillager_outpost"
                    elif building_type in ["town_hall", "center"]:
                        mc_building["minecraft_structure"] = "village/plains/town_centers/plains_fountain_01"
                    else:
                        mc_building["minecraft_structure"] = "village/plains/houses/plains_small_house_1"  # 默认值
                    
                    minecraft_buildings.append(mc_building)
                
                minecraft_map_data["layers"]["buildings"] = minecraft_buildings
        
        # 添加Minecraft特定元数据
        minecraft_map_data["minecraft_metadata"] = {
            "version": "1.18",
            "block_palette": self._generate_block_palette(),
            "style": "minecraft"
        }
        
        # 为了兼容性，添加一个get_layer方法
        def get_layer(name):
            return minecraft_map_data["layers"].get(name)
        
        minecraft_map_data["get_layer"] = get_layer
        
        return minecraft_map_data
    
    def _generate_block_palette(self):
        """根据生物群系生成Minecraft方块调色板"""
        palette = {}
        
        biome_map = self.map_data.get_layer("biome")
        if biome_map is not None:
            unique_biomes = np.unique(biome_map)
            for biome_id in unique_biomes:
                if biome_id == 0:  # 平原
                    palette[int(biome_id)] = {
                        "top": "grass_block",
                        "middle": "dirt",
                        "bottom": "stone",
                        "underwater": "sand",
                        "liquid": "water"
                    }
                elif biome_id == 1:  # 沙漠
                    palette[int(biome_id)] = {
                        "top": "sand",
                        "middle": "sandstone",
                        "bottom": "stone",
                        "underwater": "sand",
                        "liquid": "water"
                    }
                elif biome_id == 2:  # 森林
                    palette[int(biome_id)] = {
                        "top": "grass_block",
                        "middle": "dirt",
                        "bottom": "stone",
                        "underwater": "dirt",
                        "liquid": "water"
                    }
                elif biome_id == 3:  # 山脉
                    palette[int(biome_id)] = {
                        "top": "stone",
                        "middle": "stone",
                        "bottom": "deepslate",
                        "underwater": "gravel",
                        "liquid": "water"
                    }
                elif biome_id == 4:  # 海洋
                    palette[int(biome_id)] = {
                        "top": "sand",
                        "middle": "sand",
                        "bottom": "stone",
                        "underwater": "sand",
                        "liquid": "water"
                    }
                elif biome_id == 6:  # 雪地
                    palette[int(biome_id)] = {
                        "top": "snow_block",
                        "middle": "dirt",
                        "bottom": "stone",
                        "underwater": "gravel",
                        "liquid": "water"
                    }
                else:
                    # 未知生物群系的默认调色板
                    palette[int(biome_id)] = {
                        "top": "grass_block",
                        "middle": "dirt",
                        "bottom": "stone",
                        "underwater": "sand",
                        "liquid": "water"
                    }
        
        return palette
    
    def get_preview_image(self, width=512, height=512):
        """生成Minecraft风格预览图像"""
        height_map = self.map_data.get_layer("height")
        biome_map = self.map_data.get_layer("biome")
        
        if height_map is None or biome_map is None:
            img = Image.new('RGB', (width, height), (100, 100, 100))
            return img
        
        if height_map.shape != (height, width):
            from scipy.ndimage import zoom
            h_ratio = height / height_map.shape[0]
            w_ratio = width / height_map.shape[1]
            height_map = zoom(height_map, (h_ratio, w_ratio), order=0)
            biome_map = zoom(biome_map, (h_ratio, w_ratio), order=0)
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        norm_height = height_map.copy()
        if np.max(norm_height) > 0:
            norm_height = norm_height / np.max(norm_height)
        
        for y in range(height):
            for x in range(width):
                biome_id = int(biome_map[y, x])
                if biome_id in self.BIOME_COLORS:
                    base_color = self.BIOME_COLORS[biome_id]
                else:
                    base_color = self.BIOME_COLORS["default"]
                
                shade = 0.7 + 0.3 * norm_height[y, x]
                r = int(base_color[0] * shade)
                g = int(base_color[1] * shade)
                b = int(base_color[2] * shade)
                
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                img_array[y, x] = [r, g, b]
        
        img = Image.fromarray(img_array, 'RGB')
        block_size = max(1, min(height, width) // 256)
        if block_size > 1:
            small_img = img.resize((width // block_size, height // block_size), Image.Resampling.NEAREST)
            img = small_img.resize((width, height), Image.Resampling.NEAREST)
        
        return img


def get_available_styles():
    """获取可用地图样式转换器"""
    return {
        "minecraft": MinecraftStyleTransformer,
        "default": None  # 无转换
    }