# enhanced_3d_export.py

import os
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
import math
import shutil
import time
import random

# 3D模型常量配置
MODEL_LIBRARY = {
    "tree": {
        "model": "assets/models/tree.obj",
        "scale": 2.0,
        "offset_y": 0.0
    },
    "pine": {
        "model": "assets/models/pine.obj",
        "scale": 1.8,
        "offset_y": 0.0
    },
    "house": {
        "model": "assets/models/house.obj",
        "scale": 3.0,
        "offset_y": 0.0
    },
    "cave": {
        "model": "assets/models/cave_entrance.obj",
        "scale": 4.0,
        "offset_y": -1.0
    },
    "Predator": {
        "model": "assets/models/predator.obj",
        "scale": 1.5,
        "offset_y": 1.0
    },
    "Prey": {
        "model": "assets/models/prey.obj", 
        "scale": 1.0,
        "offset_y": 0.5
    }
}

# 材质配置
MATERIAL_LIBRARY = {
    "grass": {
        "diffuse": "assets/textures/grass.png",
        "normal": "assets/textures/grass_normal.png",
        "roughness": 0.7,
        "metallic": 0.0
    },
    "sand": {
        "diffuse": "assets/textures/sand.png",
        "normal": "assets/textures/sand_normal.png", 
        "roughness": 0.8,
        "metallic": 0.0
    },
    "rock": {
        "diffuse": "assets/textures/rock.png",
        "normal": "assets/textures/rock_normal.png",
        "roughness": 0.9, 
        "metallic": 0.1
    },
    "snow": {
        "diffuse": "assets/textures/snow.png",
        "normal": "assets/textures/snow_normal.png",
        "roughness": 0.3,
        "metallic": 0.0
    },
    "water": {
        "diffuse": "assets/textures/water.png",
        "normal": "assets/textures/water_normal.png",
        "roughness": 0.1,
        "metallic": 0.0,
        "transparent": True
    }
}

def ensure_asset_dirs():
    """确保资源目录存在"""
    dirs = ["export", "export/models", "export/textures", "export/materials", "export/unity", "export/unreal"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def create_heightmap_texture(height_map, filename="export/textures/heightmap.png"):
    """创建高度图纹理"""
    h = len(height_map)
    w = len(height_map[0])
    
    # 归一化高度值到0-255
    min_height = min(min(row) for row in height_map)
    max_height = max(max(row) for row in height_map)
    range_height = max_height - min_height
    
    img = Image.new('L', (w, h))
    pixels = img.load()
    
    for j in range(h):
        for i in range(w):
            normalized = int(((height_map[j][i] - min_height) / range_height) * 255)
            pixels[i, j] = normalized
    
    img.save(filename)
    return filename

def create_normal_map(height_map, filename="export/textures/normal_map.png"):
    """从高度图生成法线贴图"""
    h = len(height_map)
    w = len(height_map[0])
    
    normal_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 计算梯度
            dx = height_map[y][x+1] - height_map[y][x-1]
            dy = height_map[y+1][x] - height_map[y-1][x]
            
            # 法向量
            normal = np.array([-dx, -dy, 2.0])
            normal = normal / np.sqrt(np.sum(normal**2))
            
            # 转换到0-255范围
            normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
            normal_map[y, x] = normal
    
    img = Image.fromarray(normal_map)
    img.save(filename)
    return filename

def create_splat_map(biome_map, filename="export/textures/splat_map.png"):
    """创建材质混合贴图"""
    h = len(biome_map)
    w = len(biome_map[0])
    
    # 收集所有生物群落类型
    biome_types = set()
    for row in biome_map:
        for biome in row:
            biome_types.add(biome["name"])
    
    # 为每种生物群落分配通道
    biome_channels = {}
    for i, biome_type in enumerate(biome_types):
        channel = i % 3  # 限制为RGB三个通道
        if channel not in biome_channels:
            biome_channels[channel] = []
        biome_channels[channel].append(biome_type)
    
    # 创建混合贴图
    splat = np.zeros((h, w, 3), dtype=np.uint8)
    
    for j in range(h):
        for i in range(w):
            biome_name = biome_map[j][i]["name"]
            for channel, biomes in biome_channels.items():
                if biome_name in biomes:
                    splat[j, i, channel] = 255
    
    img = Image.fromarray(splat)
    img.save(filename)
    
    # 记录生物群落到通道的映射
    mapping = {biome: channel for channel, biomes in biome_channels.items() for biome in biomes}
    
    return filename, mapping

def create_enhanced_terrain_mesh(height_map, filename="export/models/terrain.obj"):
    """创建增强的地形网格，包含法线和UV"""
    h = len(height_map)
    w = len(height_map[0])
    
    vertices = []
    normals = []
    uvs = []
    faces = []
    
    # 生成顶点和UV
    for j in range(h):
        for i in range(w):
            vertices.append([i, j, height_map[j][i]])
            uvs.append([i/(w-1), j/(h-1)])
    
    # 计算法线
    for j in range(h):
        for i in range(w):
            nx, ny, nz = 0, 0, 1  # 默认法线朝上
            
            if i > 0 and i < w-1 and j > 0 and j < h-1:
                dx = height_map[j][i+1] - height_map[j][i-1]
                dy = height_map[j+1][i] - height_map[j-1][i]
                magnitude = math.sqrt(dx*dx + dy*dy + 4)
                nx, ny, nz = -dx/magnitude, -dy/magnitude, 2/magnitude
            
            normals.append([nx, ny, nz])
    
    # 生成面
    for j in range(h-1):
        for i in range(w-1):
            v1 = j*w + i
            v2 = j*w + (i+1)
            v3 = (j+1)*w + i
            v4 = (j+1)*w + (i+1)
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_normals=np.array(normals),
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs))
    )
    
    # 导出为OBJ
    mesh.export(filename)
    return filename

def create_material_files(biome_mapping, sea_level):
    """创建材质文件"""
    materials = {}
    
    # 为每种生物群落创建材质
    for biome_name, channel in biome_mapping.items():
        if biome_name == "Ocean":
            materials[biome_name] = MATERIAL_LIBRARY["water"]
        elif biome_name == "Beach" or biome_name == "Desert":
            materials[biome_name] = MATERIAL_LIBRARY["sand"]
        elif biome_name == "Mountain" or biome_name == "Volcano":
            materials[biome_name] = MATERIAL_LIBRARY["rock"]
        elif biome_name == "SnowPeak":
            materials[biome_name] = MATERIAL_LIBRARY["snow"]
        else:
            materials[biome_name] = MATERIAL_LIBRARY["grass"]
    
    # 写入材质文件
    with open("export/materials/terrain_materials.json", "w") as f:
        json.dump({
            "materials": materials,
            "biome_mapping": biome_mapping,
            "sea_level": sea_level
        }, f, indent=2)
    
    return "export/materials/terrain_materials.json"

def place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout):
    """放置3D模型"""
    models = []
    
    # 放置植被
    for veg in vegetation:
        x, y, veg_type = veg
        if veg_type not in MODEL_LIBRARY:
            veg_type = "tree"  # 默认使用树
        
        model_info = MODEL_LIBRARY[veg_type]
        
        models.append({
            "type": veg_type,
            "position": [x, y, height_map[y][x] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"] * np.random.uniform(0.8, 1.2),
            "model_path": model_info["model"]
        })
    
    # 放置建筑
    for bld in buildings:
        x, y, bld_type = bld
        if bld_type not in MODEL_LIBRARY:
            bld_type = "house"  # 默认使用房子
        
        model_info = MODEL_LIBRARY[bld_type]
        
        models.append({
            "type": bld_type,
            "position": [x, y, height_map[y][x] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"],
            "model_path": model_info["model"]
        })
    
    # 放置洞穴
    for cx, cy in caves:
        model_info = MODEL_LIBRARY["cave"]
        models.append({
            "type": "cave",
            "position": [cx, cy, height_map[cy][cx] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"],
            "model_path": model_info["model"]
        })
    
    # 放置生物
    if "creatures" in content_layout:
        for creature in content_layout["creatures"]:
            x, y = creature["x"], creature["y"]
            role = creature.get("role", "Prey")
            
            model_info = MODEL_LIBRARY.get(role, MODEL_LIBRARY["Prey"])
            
            models.append({
                "type": role,
                "position": [x, y, height_map[y][x] + model_info["offset_y"]],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": model_info["scale"],
                "model_path": model_info["model"],
                "attributes": creature.get("attributes", {})
            })
    
    # 保存模型信息
    with open("export/models/placed_models.json", "w") as f:
        json.dump(models, f, indent=2)
    
    return models

def create_unity_package(height_map, biome_map, models, biome_materials, content_layout, rivers, roads):
    """创建Unity包"""
    unity_dir = "export/unity"
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator", exist_ok=True)
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Models", exist_ok=True)
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Textures", exist_ok=True)
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Materials", exist_ok=True)
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Prefabs", exist_ok=True)
    os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Scripts", exist_ok=True)
    
    # 复制模型和纹理
    shutil.copy("export/models/terrain.obj", f"{unity_dir}/Assets/TerrainGenerator/Models/")
    shutil.copy("export/textures/heightmap.png", f"{unity_dir}/Assets/TerrainGenerator/Textures/")
    shutil.copy("export/textures/normal_map.png", f"{unity_dir}/Assets/TerrainGenerator/Textures/")
    shutil.copy("export/textures/splat_map.png", f"{unity_dir}/Assets/TerrainGenerator/Textures/")
    
    # 创建Unity地形设置
    with open(f"{unity_dir}/Assets/TerrainGenerator/terrain_settings.json", "w") as f:
        json.dump({
            "width": len(height_map[0]),
            "height": len(height_map),
            "heightScale": 1.0,
            "heightmapResolution": len(height_map),
            "detailResolution": 1024,
            "controlTextureResolution": 1024,
            "baseTextureResolution": 1024,
            "seaLevel": biome_materials.get("sea_level", 15.0)
        }, f, indent=2)
    
    # 创建模型放置信息
    with open(f"{unity_dir}/Assets/TerrainGenerator/model_placements.json", "w") as f:
        json.dump(models, f, indent=2)
    
    # 创建生物和事件数据
    with open(f"{unity_dir}/Assets/TerrainGenerator/game_content.json", "w") as f:
        json.dump({
            "creatures": content_layout.get("creatures", []),
            "story_events": content_layout.get("story_events", []),
            "story_overview": content_layout.get("story_overview", ""),
            "map_emotion": content_layout.get("map_emotion", {}),
            "region_emotions": content_layout.get("region_emotions", {})
        }, f, indent=2)
    
    # 创建Unity场景导入脚本
    with open(f"{unity_dir}/Assets/TerrainGenerator/Scripts/TerrainImporter.cs", "w") as f:
        f.write("""
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using UnityEditor;

#if UNITY_EDITOR
public class TerrainImporter : EditorWindow {
    [MenuItem("Tools/Import Generated Terrain")]
    static void Init() {
        TerrainImporter window = GetWindow<TerrainImporter>();
        window.Show();
    }

    void OnGUI() {
        if(GUILayout.Button("Import Terrain and Assets")) {
            ImportAll();
        }
    }

    void ImportAll() {
        // 读取配置文件
        string settingsPath = "Assets/TerrainGenerator/terrain_settings.json";
        string jsonText = File.ReadAllText(settingsPath);
        var settings = JsonUtility.FromJson<TerrainSettings>(jsonText);
        
        // 创建地形
        TerrainData terrainData = new TerrainData();
        terrainData.heightmapResolution = settings.heightmapResolution;
        terrainData.size = new Vector3(settings.width, settings.heightScale, settings.height);
        
        // 加载高度图
        Texture2D heightmap = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/TerrainGenerator/Textures/heightmap.png");
        float[,] heights = new float[settings.heightmapResolution, settings.heightmapResolution];
        
        // 设置高度图
        for (int y = 0; y < settings.heightmapResolution; y++) {
            for (int x = 0; x < settings.heightmapResolution; x++) {
                float normX = (float)x / settings.heightmapResolution;
                float normY = (float)y / settings.heightmapResolution;
                Color pixelColor = heightmap.GetPixel(
                    Mathf.FloorToInt(normX * heightmap.width),
                    Mathf.FloorToInt(normY * heightmap.height)
                );
                heights[y, x] = pixelColor.grayscale;
            }
        }
        terrainData.SetHeights(0, 0, heights);
        
        // 保存地形数据
        AssetDatabase.CreateAsset(terrainData, "Assets/TerrainGenerator/TerrainData.asset");
        
        // 创建地形GameObject
        GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
        terrainGO.name = "GeneratedTerrain";
        
        // 放置模型
        ImportModels();
        
        Debug.Log("Terrain import complete!");
    }
    
    void ImportModels() {
        string modelDataPath = "Assets/TerrainGenerator/model_placements.json";
        string jsonText = File.ReadAllText(modelDataPath);
        ModelPlacementData placementData = JsonUtility.FromJson<ModelPlacementData>(jsonText);
        
        foreach (var model in placementData.models) {
            // 加载模型预制体
            string modelPath = "Assets/TerrainGenerator/Models/" + Path.GetFileName(model.model_path);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
            
            if (prefab != null) {
                GameObject instance = Instantiate(prefab);
                instance.transform.position = new Vector3(model.position[0], model.position[2], model.position[1]);
                instance.transform.eulerAngles = new Vector3(model.rotation[0], model.rotation[1], model.rotation[2]);
                instance.transform.localScale = Vector3.one * model.scale;
                instance.name = model.type;
            }
        }
    }
}

[System.Serializable]
public class TerrainSettings {
    public int width;
    public int height;
    public float heightScale;
    public int heightmapResolution;
    public int detailResolution;
    public int controlTextureResolution;
    public int baseTextureResolution;
    public float seaLevel;
}

[System.Serializable]
public class ModelPlacement {
    public string type;
    public float[] position;
    public float[] rotation;
    public float scale;
    public string model_path;
}

[System.Serializable]
public class ModelPlacementData {
    public List<ModelPlacement> models;
}
#endif
        """)
    
    # 创建README.md
    with open(f"{unity_dir}/README.md", "w") as f:
        f.write("""
# Generated Unity Terrain Package

This package contains a procedurally generated terrain with models, materials and game content.

## Import Instructions

1. Create a new Unity project or open an existing one
2. Copy the contents of this directory into your Assets folder
3. Open the Unity Editor
4. Go to Tools > Import Generated Terrain
5. Click "Import Terrain and Assets"
6. The terrain will be generated with all models placed as specified

## Content

- Complete 3D terrain with height data
- Materials and textures
- Placed 3D models for vegetation, buildings, and other features
- Game content data including creature attributes and story events
- Rivers, roads, and other path systems

## Game Content

Check the `Assets/TerrainGenerator/game_content.json` file for:
- Creature data with attributes
- Story events with descriptions
- Emotional mapping of different regions
- Overall story overview
        """)
    
    print(f"Unity package created in '{unity_dir}'")
    return unity_dir

def create_unreal_package(height_map, biome_map, models, biome_materials, content_layout, rivers, roads):
    """创建Unreal包"""
    unreal_dir = "export/unreal"
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator", exist_ok=True)
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Models", exist_ok=True)
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Textures", exist_ok=True)
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Materials", exist_ok=True)
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Maps", exist_ok=True)
    os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Blueprints", exist_ok=True)
    
    # 复制模型和纹理
    shutil.copy("export/models/terrain.obj", f"{unreal_dir}/Content/TerrainGenerator/Models/")
    shutil.copy("export/textures/heightmap.png", f"{unreal_dir}/Content/TerrainGenerator/Textures/")
    shutil.copy("export/textures/normal_map.png", f"{unreal_dir}/Content/TerrainGenerator/Textures/")
    shutil.copy("export/textures/splat_map.png", f"{unreal_dir}/Content/TerrainGenerator/Textures/")
    
    # 创建Unreal地形配置
    with open(f"{unreal_dir}/Content/TerrainGenerator/terrain_config.json", "w") as f:
        json.dump({
            "TerrainWidth": len(height_map[0]) * 100,  # 转换为Unreal单位
            "TerrainHeight": len(height_map) * 100,
            "HeightScale": 100.0,
            "NumSections": 1,
            "SectionsPerComponent": 1,
            "QuadsPerSection": len(height_map),
            "MaxLOD": 8
        }, f, indent=2)
    
    # 创建模型放置信息
    with open(f"{unreal_dir}/Content/TerrainGenerator/model_placements.json", "w") as f:
        json.dump(models, f, indent=2)
    
    # 创建生物和事件数据
    with open(f"{unreal_dir}/Content/TerrainGenerator/game_content.json", "w") as f:
        json.dump({
            "creatures": content_layout.get("creatures", []),
            "story_events": content_layout.get("story_events", []),
            "story_overview": content_layout.get("story_overview", ""),
            "map_emotion": content_layout.get("map_emotion", {}),
            "region_emotions": content_layout.get("region_emotions", {})
        }, f, indent=2)
    
    # 创建Unreal导入脚本
    with open(f"{unreal_dir}/Content/TerrainGenerator/ImportScript.py", "w") as f:
        f.write("""
import unreal
import json
import os

def import_terrain():
    # 读取配置
    with open(os.path.join(unreal.Paths.project_content_dir(), "TerrainGenerator/terrain_config.json"), 'r') as config_file:
        config = json.load(config_file)
    
    # 创建Heightfield
    heightfield_asset = unreal.LandscapeEditorUtils.create_landscape_from_heightmap(
        unreal.Paths.project_content_dir() + "/TerrainGenerator/Textures/heightmap.png",
        config["TerrainWidth"],
        config["TerrainHeight"],
        config["NumSections"],
        config["SectionsPerComponent"],
        config["QuadsPerSection"]
    )
    
    # 导入模型
    with open(os.path.join(unreal.Paths.project_content_dir(), "TerrainGenerator/model_placements.json"), 'r') as models_file:
        model_data = json.load(models_file)
    
    for model in model_data:
        task = unreal.AssetImportTask()
        task.filename = model["model_path"]
        task.destination_path = "/Game/TerrainGenerator/Models"
        task.replace_existing = True
        task.automated = True
        
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        
        asset_path = task.get_imported_object_paths()[0]
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
            unreal.load_asset(asset_path),
            unreal.Vector(model["position"][0] * 100, model["position"][1] * 100, model["position"][2] * 100)
        )
        actor.set_actor_scale3d(unreal.Vector(model["scale"], model["scale"], model["scale"]))
        actor.set_actor_rotation(
            unreal.Rotator(model["rotation"][0], model["rotation"][1], model["rotation"][2]),
            False
        )
    
    print("Terrain and models imported successfully!")

# Run the import script
import_terrain()
        """)
    
    # 创建README.md
    with open(f"{unreal_dir}/README.md", "w") as f:
        f.write("""
# Generated Unreal Engine Terrain Package

This package contains a procedurally generated terrain with models, materials and game content.

## Import Instructions

1. Create a new Unreal Engine project or open an existing one
2. Copy the contents of this directory into your Content folder
3. Open the Unreal Editor
4. Go to Window > Developer Tools > Python Console
5. Run the following command:
   ```python
   exec(open("/Game/TerrainGenerator/ImportScript.py").read())
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   



























































# 添加3D模型配置字典（从enhanced_3d_export.py）
MODEL_LIBRARY = {
    "tree": {
        "model": "assets/models/tree.obj",
        "scale": 2.0,
        "offset_y": 0.0
    },
    "pine": {
        "model": "assets/models/pine.obj",
        "scale": 1.8,
        "offset_y": 0.0
    },
    "house": {
        "model": "assets/models/house.obj",
        "scale": 3.0,
        "offset_y": 0.0
    },
    "cave": {
        "model": "assets/models/cave_entrance.obj",
        "scale": 4.0,
        "offset_y": -1.0
    },
    "Predator": {
        "model": "assets/models/predator.obj",
        "scale": 1.5,
        "offset_y": 1.0
    },
    "Prey": {
        "model": "assets/models/prey.obj", 
        "scale": 1.0,
        "offset_y": 0.5
    }
}

# 材质配置（从enhanced_3d_export.py）
MATERIAL_LIBRARY = {
    "grass": {
        "diffuse": "assets/textures/grass.png",
        "normal": "assets/textures/grass_normal.png",
        "roughness": 0.7,
        "metallic": 0.0
    },
    "sand": {
        "diffuse": "assets/textures/sand.png",
        "normal": "assets/textures/sand_normal.png", 
        "roughness": 0.8,
        "metallic": 0.0
    },
    "rock": {
        "diffuse": "assets/textures/rock.png",
        "normal": "assets/textures/rock_normal.png",
        "roughness": 0.9, 
        "metallic": 0.1
    },
    "snow": {
        "diffuse": "assets/textures/snow.png",
        "normal": "assets/textures/snow_normal.png",
        "roughness": 0.3,
        "metallic": 0.0
    },
    "water": {
        "diffuse": "assets/textures/water.png",
        "normal": "assets/textures/water_normal.png",
        "roughness": 0.1,
        "metallic": 0.0,
        "transparent": True
    }
}

def ensure_asset_dirs():
    """确保资源目录存在"""
    dirs = ["export", "export/models", "export/textures", "export/materials", "export/unity", "export/unreal"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def create_heightmap_texture(height_map, filename="export/textures/heightmap.png"):
    """创建高度图纹理"""
    h = len(height_map)
    w = len(height_map[0])
    
    # 归一化高度值到0-255
    min_height = min(min(row) for row in height_map)
    max_height = max(max(row) for row in height_map)
    range_height = max_height - min_height
    
    img = Image.new('L', (w, h))
    pixels = img.load()
    
    for j in range(h):
        for i in range(w):
            normalized = int(((height_map[j][i] - min_height) / range_height) * 255)
            pixels[i, j] = normalized
    
    img.save(filename)
    return filename

def create_normal_map(height_map, filename="export/textures/normal_map.png"):
    """从高度图生成法线贴图"""
    h = len(height_map)
    w = len(height_map[0])
    
    normal_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 计算梯度
            dx = height_map[y][x+1] - height_map[y][x-1]
            dy = height_map[y+1][x] - height_map[y-1][x]
            
            # 法向量
            normal = np.array([-dx, -dy, 2.0])
            normal = normal / np.sqrt(np.sum(normal**2))
            
            # 转换到0-255范围
            normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
            normal_map[y, x] = normal
    
    img = Image.fromarray(normal_map)
    img.save(filename)
    return filename

def create_splat_map(biome_map, filename="export/textures/splat_map.png"):
    """创建材质混合贴图"""
    h = len(biome_map)
    w = len(biome_map[0])
    
    # 收集所有生物群落类型
    biome_types = set()
    for row in biome_map:
        for biome in row:
            biome_types.add(biome["name"])
    
    # 为每种生物群落分配通道
    biome_channels = {}
    for i, biome_type in enumerate(biome_types):
        channel = i % 3  # 限制为RGB三个通道
        if channel not in biome_channels:
            biome_channels[channel] = []
        biome_channels[channel].append(biome_type)
    
    # 创建混合贴图
    splat = np.zeros((h, w, 3), dtype=np.uint8)
    
    for j in range(h):
        for i in range(w):
            biome_name = biome_map[j][i]["name"]
            for channel, biomes in biome_channels.items():
                if biome_name in biomes:
                    splat[j, i, channel] = 255
    
    img = Image.fromarray(splat)
    img.save(filename)
    
    # 记录生物群落到通道的映射
    mapping = {biome: channel for channel, biomes in biome_channels.items() for biome in biomes}
    
    return filename, mapping

def create_enhanced_terrain_mesh(height_map, filename="export/models/terrain.obj"):
    """创建增强的地形网格，包含法线和UV"""
    h = len(height_map)
    w = len(height_map[0])
    
    vertices = []
    normals = []
    uvs = []
    faces = []
    
    # 生成顶点和UV
    for j in range(h):
        for i in range(w):
            vertices.append([i, j, height_map[j][i]])
            uvs.append([i/(w-1), j/(h-1)])
    
    # 计算法线
    for j in range(h):
        for i in range(w):
            nx, ny, nz = 0, 0, 1  # 默认法线朝上
            
            if i > 0 and i < w-1 and j > 0 and j < h-1:
                dx = height_map[j][i+1] - height_map[j][i-1]
                dy = height_map[j+1][i] - height_map[j-1][i]
                magnitude = math.sqrt(dx*dx + dy*dy + 4)
                nx, ny, nz = -dx/magnitude, -dy/magnitude, 2/magnitude
            
            normals.append([nx, ny, nz])
    
    # 生成面
    for j in range(h-1):
        for i in range(w-1):
            v1 = j*w + i
            v2 = j*w + (i+1)
            v3 = (j+1)*w + i
            v4 = (j+1)*w + (i+1)
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_normals=np.array(normals),
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs))
    )
    
    # 导出为OBJ
    mesh.export(filename)
    return filename

def create_material_files(biome_mapping, sea_level):
    """创建材质文件"""
    materials = {}
    
    # 为每种生物群落创建材质
    for biome_name, channel in biome_mapping.items():
        if biome_name == "Ocean":
            materials[biome_name] = MATERIAL_LIBRARY["water"]
        elif biome_name == "Beach" or biome_name == "Desert":
            materials[biome_name] = MATERIAL_LIBRARY["sand"]
        elif biome_name == "Mountain" or biome_name == "Volcano":
            materials[biome_name] = MATERIAL_LIBRARY["rock"]
        elif biome_name == "SnowPeak":
            materials[biome_name] = MATERIAL_LIBRARY["snow"]
        else:
            materials[biome_name] = MATERIAL_LIBRARY["grass"]
    
    # 写入材质文件
    with open("export/materials/terrain_materials.json", "w") as f:
        json.dump({
            "materials": materials,
            "biome_mapping": biome_mapping,
            "sea_level": sea_level
        }, f, indent=2)
    
    return "export/materials/terrain_materials.json"

def place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout):
    """放置3D模型"""
    models = []
    
    # 放置植被
    for veg in vegetation:
        x, y, veg_type = veg
        if veg_type not in MODEL_LIBRARY:
            veg_type = "tree"  # 默认使用树
        
        model_info = MODEL_LIBRARY[veg_type]
        
        models.append({
            "type": veg_type,
            "position": [x, y, height_map[y][x] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"] * np.random.uniform(0.8, 1.2),
            "model_path": model_info["model"]
        })
    
    # 放置建筑
    for bld in buildings:
        x, y, bld_type = bld
        if bld_type not in MODEL_LIBRARY:
            bld_type = "house"  # 默认使用房子
        
        model_info = MODEL_LIBRARY[bld_type]
        
        models.append({
            "type": bld_type,
            "position": [x, y, height_map[y][x] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"],
            "model_path": model_info["model"]
        })
    
    # 放置洞穴
    for cx, cy in caves:
        model_info = MODEL_LIBRARY["cave"]
        models.append({
            "type": "cave",
            "position": [cx, cy, height_map[cy][cx] + model_info["offset_y"]],
            "rotation": [0, np.random.uniform(0, 360), 0],
            "scale": model_info["scale"],
            "model_path": model_info["model"]
        })
    
    # 放置生物
    if "creatures" in content_layout:
        for creature in content_layout["creatures"]:
            x, y = creature["x"], creature["y"]
            role = creature.get("role", "Prey")
            
            model_info = MODEL_LIBRARY.get(role, MODEL_LIBRARY["Prey"])
            
            models.append({
                "type": role,
                "position": [x, y, height_map[y][x] + model_info["offset_y"]],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": model_info["scale"],
                "model_path": model_info["model"],
                "attributes": creature.get("attributes", {})
            })
    
    # 保存模型信息
    with open("export/models/placed_models.json", "w") as f:
        json.dump(models, f, indent=2)
    
    return models

@dataclass
class MapExportConfig:
    """导出配置，支持高度定制化的导出参数"""
    output_dir: str = "./exports"
    base_filename: str = "game_map"
    texture_size: Tuple[int, int] = (2048, 2048)
    generate_textures: bool = True
    include_metadata: bool = True
    compress_output: bool = False
    export_normals: bool = True
    export_heightmap: bool = True
    level_of_detail: int = 1  # 1=全分辨率，2=半分辨率，以此类推
    memory_efficient: bool = True  # 使用流式处理减少内存占用
    multithreaded: bool = True
    max_workers: int = 4
    # 新增引擎专用参数
    unity_export_version: str = "2022.3"  # Unity目标版本
    unreal_export_version: str = "5.3"    # Unreal目标版本
    export_collision: bool = True         # 是否导出碰撞数据
    lightmap_uvs: bool = False            # 是否生成光照贴图UV
  
class JSONExporterBase:
    """JSON导出器基类"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        self.config = config or MapExportConfig()
        self._cached_data = None  # 用于内存高效模式
    
    def _prepare_export_data(self, map_data: Dict) -> Dict:
        """准备导出数据（模板方法）"""
        # 创建实例再调用方法，而不是直接通过类调用
        normalizer = MapDataNormalizer()
        normalized_data = normalizer.prepare_map_data(map_data)
        
        # 应用LOD
        if self.config.level_of_detail > 1:
            normalized_data = self._apply_lod(normalized_data)
        
        # 流式处理时不清除缓存
        if not self.config.memory_efficient:
            self._cached_data = normalized_data
        
        return normalized_data
    
    def _apply_lod(self, data: Dict) -> Dict:
        """应用细节层次"""
        lod_step = self.config.level_of_detail
        h = data["height"]
        w = data["width"]
        
        # 处理高度图
        data["height_map"] = [
            [row[i] for i in range(0, w, lod_step)]
            for row in data["height_map"][::lod_step]
        ]
        
        # 更新尺寸
        data["width"] = len(data["height_map"][0]) if data["height_map"] else 0
        data["height"] = len(data["height_map"])
        
        # 处理其他图层
        for layer in ["biome_map", "rivers"]:
            data[layer] = [
                [row[i] for i in range(0, w, lod_step)]
                for row in data[layer][::lod_step]
            ]
        
        # 处理实体坐标
        def adjust_coords(coords):
            return [(x//lod_step, y//lod_step) for x, y in coords]
        
        data["road_coords"] = adjust_coords(data["road_coords"])
        data["caves"] = [adjust_coords(cave) for cave in data["caves"]]
        
        return data
    
    def _generate_metadata(self, data: Dict) -> Dict:
        """生成元数据"""
        if not self.config.include_metadata:
            return {}
        
        return {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "stats": {
                "vertex_count": data["width"] * data["height"],
                "entity_counts": {
                    "vegetation": len(data["vegetation"]),
                    "buildings": len(data["buildings"]),
                    "caves": sum(len(c) for c in data["caves"])
                }
            }
        }
        
    def _validate_data(self, data: Dict) -> bool:
        """验证导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        required_fields = ["width", "height", "height_map"]
        for field in required_fields:
            if field not in data:
                self.logger.error(f"导出数据缺少必要字段: {field}")
                return False
                
        # 验证尺寸
        if data["width"] <= 0 or data["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {data['width']}x{data['height']}")
            return False
            
        return True
    
    def _write_json_file(self, data: Dict, filename: str) -> bool:
        """将数据写入JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 写入JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"成功写入JSON文件: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"写入JSON文件失败: {str(e)}")
            return False
    
    def _generate_texture_path(self) -> str:
        """生成纹理文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.base_filename}_texture_{timestamp}.png"
        return os.path.join(self.config.output_dir, filename)
    
    def _process_biomes(self, biome_map: List[List[Dict]]) -> Dict:
        """处理生物群系数据"""
        biome_types = set()
        biome_counts = {}
        
        # 统计生物群系类型和数量
        for row in biome_map:
            for biome in row:
                biome_type = biome.get("name", "unknown")
                biome_types.add(biome_type)
                biome_counts[biome_type] = biome_counts.get(biome_type, 0) + 1
        
        return {
            "types": list(biome_types),
            "counts": biome_counts,
            "map": biome_map
        }
    
    def _process_entities(self, entities: List[Dict]) -> List[Dict]:
        """处理实体数据"""
        processed = []
        
        for entity in entities:
            # 创建副本以避免修改原数据
            processed_entity = entity.copy()
            
            # 添加元数据
            if "type" in entity:
                entity_type = entity["type"]
                processed_entity["prefab"] = f"Prefabs/{entity_type}"
                
            processed.append(processed_entity)
            
        return processed
    
    def _process_rivers(self, rivers: List[List[bool]]) -> Dict:
        """处理河流数据"""
        width = len(rivers[0]) if rivers and rivers[0] else 0
        height = len(rivers)
        
        # 提取河流坐标
        river_coords = []
        for y in range(height):
            for x in range(width):
                if rivers[y][x]:
                    river_coords.append([x, y])
        
        # 计算河流网络特性
        river_segments = self._extract_river_segments(river_coords)
        
        return {
            "coordinates": river_coords,
            "segments": river_segments,
            "width": width,
            "height": height
        }
    
    def _extract_river_segments(self, river_coords: List[List[int]]) -> List[Dict]:
        """提取河流分段"""
        # 实际项目中这会更复杂，例如寻找连续的河流段
        # 这里只是一个简化的实现
        return [{"start": [0, 0], "end": [10, 10], "width": 1.0}]
    
    def _process_caves(self, caves: List[Tuple[int, int]]) -> List[Dict]:
        """处理洞穴数据"""
        processed_caves = []
        
        for i, (x, y) in enumerate(caves):
            processed_caves.append({
                "id": i,
                "position": [x, y],
                "type": "cave_entrance",
                "size": random.uniform(1.0, 3.0)
            })
        
        return processed_caves
    
    def _process_roads(self, road_coords: List[Tuple[int, int]]) -> Dict:
        """处理道路数据"""
        road_types = ["dirt", "stone", "paved"]
        processed_roads = []
        
        # 简单处理：为每个坐标分配一个随机道路类型
        for x, y in road_coords:
            road_type = random.choice(road_types)
            processed_roads.append({
                "position": [x, y],
                "type": road_type,
                "width": 1.0 if road_type == "dirt" else (1.5 if road_type == "stone" else 2.0)
            })
        
        return {
            "coordinates": road_coords,
            "processed": processed_roads
        }
        
    def _apply_lod(self, data: Dict) -> Dict:
        """应用细节层次"""
        lod_step = self.config.level_of_detail
        h = len(data["height_map"])
        w = len(data["height_map"][0]) if h > 0 else 0
        
        # 处理高度图
        data["height_map"] = [
            [row[i] for i in range(0, len(row), lod_step)]
            for row in data["height_map"][::lod_step]
        ]
        
        # 更新尺寸
        data["width"] = len(data["height_map"][0]) if data["height_map"] else 0
        data["height"] = len(data["height_map"])
        
        # 处理其他图层
        if "biome_map" in data:
            data["biome_map"] = [
                [row[i] for i in range(0, len(row), lod_step)]
                for row in data["biome_map"][::lod_step]
            ]
        
        if "rivers" in data:
            data["rivers"] = [
                [row[i] for i in range(0, len(row), lod_step)]
                for row in data["rivers"][::lod_step]
            ]
        
        # 处理实体坐标
        def adjust_coords(coords):
            return [(x//lod_step, y//lod_step) for x, y in coords]
        
        if "road_coords" in data:
            data["road_coords"] = adjust_coords(data["road_coords"])
        
        if "caves" in data:
            data["caves"] = adjust_coords(data["caves"])
        
        return data

    def _generate_metadata(self, data: Dict) -> Dict:
        """生成元数据"""
        if not self.config.include_metadata:
            return {}
        
        metadata = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "stats": {
                "map_size": {"width": data["width"], "height": data["height"]},
                "entity_counts": {}
            }
        }
        
        # 计算各类实体数量
        if "vegetation" in data:
            metadata["stats"]["entity_counts"]["vegetation"] = len(data["vegetation"])
        if "buildings" in data:
            metadata["stats"]["entity_counts"]["buildings"] = len(data["buildings"])
        if "caves" in data:
            metadata["stats"]["entity_counts"]["caves"] = len(data["caves"])
        if "road_coords" in data:
            metadata["stats"]["entity_counts"]["road_segments"] = len(data["road_coords"])
        
        # 添加生物群系统计
        if "biome_map" in data:
            biome_counts = {}
            for row in data["biome_map"]:
                for biome in row:
                    biome_name = biome["name"]
                    biome_counts[biome_name] = biome_counts.get(biome_name, 0) + 1
            metadata["stats"]["biomes"] = biome_counts
        
        return metadata

    def _validate_data(self, data: Dict) -> bool:
        """验证导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        required_fields = ["width", "height", "height_map"]
        for field in required_fields:
            if field not in data:
                self.logger.error(f"导出数据缺少必要字段: {field}")
                return False
                
        # 验证尺寸
        if data["width"] <= 0 or data["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {data['width']}x{data['height']}")
            return False
            
        # 验证高度图
        if not data["height_map"] or len(data["height_map"]) != data["height"]:
            self.logger.error(f"高度图尺寸不匹配: 期望 {data['height']} 行，实际 {len(data['height_map'])} 行")
            return False
        
        if len(data["height_map"][0]) != data["width"]:
            self.logger.error(f"高度图尺寸不匹配: 期望 {data['width']} 列，实际 {len(data['height_map'][0])} 列")
            return False
            
        return True

    def _write_json_file(self, data: Dict, filename: str) -> bool:
        """将数据写入JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 写入JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"成功写入JSON文件: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"写入JSON文件失败: {str(e)}")
            return False

    def _generate_texture_path(self) -> str:
        """生成纹理文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.base_filename}_texture_{timestamp}.png"
        return os.path.join(self.config.output_dir, filename)
#########################################
#地图格式标准化
########################################
class MapDataNormalizer:
    """地图数据标准化器 - 处理各种输入格式并转换为统一标准"""
    
    @staticmethod
    def normalize_height_map(height_map: Any) -> List[List[float]]:
        """标准化高度图为二维浮点数组"""
        try:
            # 处理numpy数组
            if isinstance(height_map, np.ndarray):
                return height_map.astype(float).tolist()
            
            # 处理嵌套列表
            if isinstance(height_map, list):
                if not height_map:
                    return []
                
                # 确保每行长度一致
                h = len(height_map)
                w = max(len(row) if isinstance(row, (list, tuple)) else 0 for row in height_map)
                result = []
                
                for j in range(h):
                    row = height_map[j]
                    if not isinstance(row, (list, tuple)):
                        row = [float(row)]
                    
                    # 填充或截断至统一宽度
                    result.append([float(row[i]) if i < len(row) else 0.0 for i in range(w)])
                
                return result
            
            logger.error(f"无法处理的高度图类型: {type(height_map)}")
            return [[0.0]]
            
        except Exception as e:
            logger.error(f"高度图标准化失败: {e}")
            return [[0.0]]
    
    @staticmethod
    def normalize_biome_map(biome_map: Any) -> List[List[Dict[str, Any]]]:
        """标准化生物群系图为二维字典数组"""
        try:
            h = len(biome_map)
            if h == 0:
                return []
            
            # 确定宽度
            w = len(biome_map[0]) if isinstance(biome_map[0], (list, tuple)) else 1
            
            result = []
            for j in range(h):
                row = []
                for i in range(w):
                    try:
                        biome = biome_map[j][i]
                        if isinstance(biome, dict):
                            # 标准化颜色数组
                            if "color" in biome:
                                color = biome["color"]
                                if isinstance(color, np.ndarray):
                                    color = color.tolist()
                                # 确保颜色为RGB三元组
                                if len(color) >= 3:
                                    color = [int(c) for c in color[:3]]
                                else:
                                    color = [128, 128, 128]  # 默认灰色
                                biome["color"] = color
                            
                            # 确保名称为字符串
                            if "name" in biome:
                                biome["name"] = str(biome["name"])
                            else:
                                biome["name"] = "unknown"
                                
                            row.append(biome)
                        else:
                            # 如果不是字典，创建默认值
                            row.append({"name": "unknown", "color": [128, 128, 128]})
                    except IndexError:
                        row.append({"name": "unknown", "color": [128, 128, 128]})
                result.append(row)
            
            return result
        except Exception as e:
            logger.error(f"生物群系标准化失败: {e}")
            return [[{"name": "error", "color": [255, 0, 0]}]]
    
    @staticmethod
    def normalize_boolean_grid(grid: Any, h: int, w: int) -> List[List[bool]]:
        """将任意格式的网格数据标准化为布尔值二维数组"""
        try:
            # 处理空值
            if grid is None:
                return [[False for _ in range(w)] for _ in range(h)]
            
            # 处理numpy数组
            if isinstance(grid, np.ndarray):
                # 处理多维数组
                if grid.ndim > 2:
                    # 对于3D+数组，如果任一维度为True则视为True
                    grid = np.any(grid, axis=tuple(range(2, grid.ndim)))
                
                # 调整尺寸
                if grid.shape != (h, w):
                    # 创建新数组，尽可能复制原始数据
                    new_grid = np.zeros((h, w), dtype=bool)
                    h_copy = min(h, grid.shape[0])
                    w_copy = min(w, grid.shape[1])
                    new_grid[:h_copy, :w_copy] = grid[:h_copy, :w_copy]
                    grid = new_grid
                
                # 转为Python列表
                return grid.tolist()
            
            # 处理嵌套列表
            if isinstance(grid, list):
                result = []
                
                # 根据输入列表决定遍历方式
                for j in range(min(h, len(grid))):
                    row = grid[j]
                    if isinstance(row, list):
                        # 处理二维列表
                        result.append([bool(row[i]) if i < len(row) else False 
                                      for i in range(w)])
                    else:
                        # 处理一维列表（视为单行）
                        result.append([bool(grid[j])] + [False] * (w - 1))
                
                # 填充缺失行
                while len(result) < h:
                    result.append([False] * w)
                
                return result
            
            # 处理其他类型（如标量值）
            return [[bool(grid) for _ in range(w)] for _ in range(h)]
            
        except Exception as e:
            logger.error(f"布尔网格标准化失败: {e}")
            return [[False for _ in range(w)] for _ in range(h)]
    
    @staticmethod
    def normalize_entity_data(entities: List, required_fields: List[str] = None) -> List[Dict]:
        """标准化实体数据（如建筑、植被）"""
        if not required_fields:
            required_fields = ["x", "y", "type"]
            
        result = []
        
        if not entities:
            return result
            
        for entity in entities:
            # 跳过无效数据
            if not entity or len(entity) < len(required_fields):
                continue
                
            # 对元组类型进行转换
            if isinstance(entity, (tuple, list)):
                try:
                    entity_dict = {
                        required_fields[i]: int(entity[i]) if i < 2 else str(entity[i])
                        for i in range(min(len(entity), len(required_fields)))
                    }
                    
                    # 确保所有必需字段存在
                    for field in required_fields:
                        if field not in entity_dict:
                            if field in ["x", "y"]:
                                entity_dict[field] = 0
                            else:
                                entity_dict[field] = "unknown"
                    
                    result.append(entity_dict)
                except (TypeError, IndexError) as e:
                    logger.warning(f"实体数据转换错误: {e}, 实体: {entity}")
                    continue
            
            # 已经是字典格式
            elif isinstance(entity, dict):
                # 复制字典避免修改原数据
                entity_dict = {}
                
                # 处理必需字段
                for field in required_fields:
                    if field in entity:
                        # 类型转换
                        if field in ["x", "y"]:
                            entity_dict[field] = int(entity[field])
                        else:
                            entity_dict[field] = str(entity[field])
                    else:
                        # 默认值
                        if field in ["x", "y"]:
                            entity_dict[field] = 0
                        else:
                            entity_dict[field] = "unknown"
                            
                # 复制其他字段
                for k, v in entity.items():
                    if k not in required_fields:
                        entity_dict[k] = v
                        
                result.append(entity_dict)
                
        return result
    
    @staticmethod
    def normalize_content_layout(content_layout: Dict) -> Dict:
        """标准化内容布局数据"""
        if not content_layout or not isinstance(content_layout, dict):
            return {}
            
        result = {}
        
        # 标准化故事事件
        if "story_events" in content_layout:
            events = []
            for event in content_layout["story_events"]:
                if isinstance(event, dict):
                    # 处理事件对象
                    event_dict = {
                        "x": int(event.get("x", 0)),
                        "y": int(event.get("y", 0)),
                        "description": str(event.get("description", ""))
                    }
                    
                    # 处理触发器
                    if "trigger" in event and event["trigger"]:
                        try:
                            trigger = event["trigger"]
                            if isinstance(trigger, (list, tuple, np.ndarray)) and len(trigger) >= 2:
                                # 从numpy数组提取值
                                if hasattr(trigger[0], 'item'):
                                    x = int(trigger[0].item())
                                else:
                                    x = int(trigger[0])
                                    
                                if hasattr(trigger[1], 'item'):
                                    y = int(trigger[1].item())
                                else:
                                    y = int(trigger[1])
                                    
                                event_dict["trigger"] = [x, y]
                            else:
                                event_dict["trigger"] = [0, 0]
                        except (IndexError, TypeError, ValueError):
                            event_dict["trigger"] = [0, 0]
                    
                    events.append(event_dict)
                    
            result["story_events"] = events
        
        # 标准化生物数据
        if "creatures" in content_layout:
            creatures = []
            for creature in content_layout["creatures"]:
                if isinstance(creature, dict):
                    # 处理生物属性
                    creature_dict = {
                        "x": int(creature.get("x", 0)),
                        "y": int(creature.get("y", 0)),
                        "role": str(creature.get("role", "unknown")),
                    }
                    
                    # 处理属性字典
                    if "attributes" in creature:
                        attributes = {}
                        for k, v in creature["attributes"].items():
                            # 转换numpy类型
                            if isinstance(v, np.number):
                                attributes[k] = float(v)
                            else:
                                attributes[k] = v
                        creature_dict["attributes"] = attributes
                        
                    creatures.append(creature_dict)
                    
            result["creatures"] = creatures
        
        # 复制其他简单字段
        for k, v in content_layout.items():
            if k not in result and k not in ["story_events", "creatures"]:
                if isinstance(v, (str, int, float, bool)):
                    result[k] = v
                elif isinstance(v, (np.integer, np.floating)):
                    result[k] = v.item()  # 转换numpy标量
                
        return result
    
    @staticmethod
    def extract_road_coords(roads, w, h) -> List[Tuple[int, int]]:
        """从各种格式提取道路坐标"""
        road_coords = []
        
        # 处理坐标列表
        if isinstance(roads, list):
            for point in roads:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    # 从各种类型提取坐标
                    try:
                        x = int(point[0])
                        y = int(point[1])
                        # 确保坐标在地图范围内
                        if 0 <= x < w and 0 <= y < h:
                            road_coords.append((x, y))
                    except (IndexError, TypeError, ValueError):
                        continue
        # 处理布尔网格
        elif isinstance(roads, np.ndarray) and roads.ndim == 2:
            h_arr, w_arr = roads.shape
            for j in range(min(h_arr, h)):
                for i in range(min(w_arr, w)):
                    if roads[j, i]:
                        road_coords.append((i, j))
        
        return road_coords
    
    def extract_cave_coords(self, caves, w, h) -> List[Tuple[int, int]]:
        """从洞穴数据提取有效坐标"""
        cave_coords = []
        
        if not caves:
            return cave_coords
        
        # 处理新格式：字典格式的洞穴数据 {"caves": [...], "entrances": [...]}
        if isinstance(caves, dict) and "caves" in caves:
            for point in caves.get("caves", []):
                if isinstance(point, dict) and "x" in point and "y" in point:
                    x, y = int(point["x"]), int(point["y"])
                    if 0 <= x < w and 0 <= y < h:
                        cave_coords.append((x, y))
            return cave_coords
                
        # 处理各种可能的洞穴数据格式（旧格式兼容）
        for cave in caves:
            # 处理numpy数组类型
            if isinstance(cave, np.ndarray):
                if cave.ndim == 2 and cave.shape[1] >= 2:
                    # 二维坐标数组
                    for i in range(cave.shape[0]):
                        try:
                            x, y = int(cave[i, 0]), int(cave[i, 1])
                            if 0 <= x < w and 0 <= y < h:
                                cave_coords.append((x, y))
                        except (IndexError, TypeError):
                            continue
                elif cave.ndim == 1 and cave.size >= 2:
                    # 一维坐标数组
                    try:
                        x, y = int(cave[0]), int(cave[1])
                        if 0 <= x < w and 0 <= y < h:
                            cave_coords.append((x, y))
                    except (IndexError, TypeError):
                        continue
            
            # 处理嵌套列表
            elif isinstance(cave, list):
                # 检查是否是点列表
                if all(isinstance(point, (list, tuple, np.ndarray)) for point in cave):
                    for point in cave:
                        try:
                            if len(point) >= 2:
                                x, y = int(point[0]), int(point[1])
                                if 0 <= x < w and 0 <= y < h:
                                    cave_coords.append((x, y))
                        except (IndexError, TypeError):
                            continue
                # 单个坐标点
                elif len(cave) >= 2:
                    try:
                        x, y = int(cave[0]), int(cave[1])
                        if 0 <= x < w and 0 <= y < h:
                            cave_coords.append((x, y))
                    except (IndexError, TypeError):
                        continue
                        
        return cave_coords

    def normalize_boolean_grid(self, grid, h, w) -> List[List[bool]]:
        """将任意格式的网格数据标准化为布尔值二维数组"""
        try:
            # 处理空值
            if grid is None:
                return [[False for _ in range(w)] for _ in range(h)]
            
            # 处理numpy数组
            if isinstance(grid, np.ndarray):
                # 处理多维数组
                if grid.ndim > 2:
                    # 对于3D+数组，如果任一维度为True则视为True
                    grid = np.any(grid, axis=tuple(range(2, grid.ndim)))
                
                # 确保值是布尔型（有可能是0/1整数）
                grid = grid.astype(bool)
                
                # 调整尺寸
                if grid.shape != (h, w):
                    # 创建新数组，尽可能复制原始数据
                    new_grid = np.zeros((h, w), dtype=bool)
                    h_copy = min(h, grid.shape[0])
                    w_copy = min(w, grid.shape[1])
                    new_grid[:h_copy, :w_copy] = grid[:h_copy, :w_copy]
                    grid = new_grid
                
                # 转为Python列表
                return grid.tolist()
            
            # 处理嵌套列表
            if isinstance(grid, list):
                result = []
                
                # 根据输入列表决定遍历方式
                for j in range(min(h, len(grid))):
                    row = grid[j]
                    if isinstance(row, list):
                        # 处理二维列表
                        result.append([bool(row[i]) if i < len(row) else False 
                                    for i in range(w)])
                    else:
                        # 处理一维列表（视为单行）
                        result.append([bool(grid[j])] + [False] * (w - 1))
                
                # 填充缺失行
                while len(result) < h:
                    result.append([False] * w)
                
                return result
            
            # 处理其他类型（如标量值）
            return [[bool(grid) for _ in range(w)] for _ in range(h)]
            
        except Exception as e:
            self.logger.error(f"布尔网格标准化失败: {e}")
            return [[False for _ in range(w)] for _ in range(h)]

    def process_road_types(self, road_network, road_types, w, h):
        """处理道路网络和道路类型数据"""
        # 确保有效的数据
        if road_types is None or not isinstance(road_types, np.ndarray):
            # 如果没有道路类型数据，则创建默认值（全部为1）
            processed_types = np.ones((h, w), dtype=int)
        else:
            # 复制输入，确保不会修改原始数据
            processed_types = road_types.copy()
            
            # 调整尺寸
            if processed_types.shape != (h, w):
                temp_types = np.ones((h, w), dtype=int)
                h_copy = min(h, processed_types.shape[0])
                w_copy = min(w, processed_types.shape[1])
                temp_types[:h_copy, :w_copy] = processed_types[:h_copy, :w_copy]
                processed_types = temp_types
        
        # 将布尔道路网络与道路类型结合
        normalized_network = self.normalize_boolean_grid(road_network, h, w)
        
        # 将道路类型转换为列表
        road_type_list = processed_types.tolist()
        
        # 确保只有在道路存在的地方才有道路类型
        for y in range(h):
            for x in range(w):
                if not normalized_network[y][x]:
                    road_type_list[y][x] = 0
        
        return {
            "network": normalized_network,
            "types": road_type_list
        }

    def prepare_map_data(self, map_data):
        """标准化地图数据包（仅修改此方法）"""
        # 修改点1：解包逻辑兼容旧版和新版元组结构
        if isinstance(map_data, tuple):
            # 处理旧版元组结构（最后一个元素是包含 roads_map 和 roads_types 的元组）
            if len(map_data) == 11:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_data) = map_data
                roads_map, roads_types = roads_data if roads_data else (None, None)
            # 处理新版元组结构（直接包含 roads_map 和 roads_types）
            elif len(map_data) >= 12:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_map, roads_types) = map_data[:12]  # 截取前12个元素
            else:
                raise ValueError("无效的输入元组结构")
        elif hasattr(map_data, 'unpack'):
            # 处理 MapData.unpack() 返回的元组（无论结构如何）
            unpacked = map_data.unpack()
            if len(unpacked) == 11:
                roads_map, roads_types = unpacked[-1]  # 旧版结构，解包嵌套元组
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads) = unpacked[:-1]
            else:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_map, roads_types) = unpacked[:12]  # 新版结构直接解包
        else:
            # 字典格式处理保持不变
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            content_layout = map_data.get("content_layout", {})
            caves = map_data.get("caves", [])
            map_params = map_data.get("params", {})
            biome_data = map_data.get("biome_data", None)
            roads = map_data.get("roads", [])
            roads_map = map_data.get("roads_map", None)
            roads_types = map_data.get("roads_types", None)

        # 计算尺寸
        if isinstance(height_map, np.ndarray):
            h, w = height_map.shape
        elif hasattr(height_map, 'shape'):
            h, w = height_map.shape[0], height_map.shape[1]
        else:
            h = len(height_map)
            w = len(height_map[0]) if h > 0 else 0

        # 添加尺寸验证
        if w <= 0 or h <= 0:
            self.logger.error(f"无效的地图尺寸: width={w}, height={h}，请检查输入的height_map")
            raise ValueError("地图尺寸无效，width或height为0")

        return {
            "width": w,
            "height": h,
            "height_map": self.normalize_height_map(height_map),
            "biome_map": self.normalize_biome_map(biome_map),
            "vegetation": self.normalize_entity_data(vegetation),
            "buildings": self.normalize_entity_data(buildings),
            "rivers": self.normalize_boolean_grid(rivers, h, w),
            "content_layout": self.normalize_content_layout(content_layout),
            "caves": self.extract_cave_coords(caves, w, h),
            "road_coords": self.extract_road_coords(roads, w, h),
            "roads_data": self.process_road_types(roads_map, roads_types, w, h) if roads_map else None,
            "map_params": map_params,
            "biome_data": biome_data
        }
###########################################
#定义地图导出器接口
###########################################
class MapExporter(ABC):
    """抽象基类，定义地图导出器接口"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        self.config = config or MapExportConfig()
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _prepare_map_data_dict(self, map_data):
        """将 MapData 对象转换为字典""" 
        if hasattr(map_data, "layers"):  # 直接检查是否为MapData对象
            width = getattr(map_data, "map_width", None)
            height = getattr(map_data, "map_height", None)
            
            # 如果width或height缺失，尝试从height_map推断
            height_map = map_data.get_layer("height")
            if (width is None or height is None) and height_map is not None:
                if isinstance(height_map, np.ndarray):
                    print("numpy数组")
                    height, width = height_map.shape
                elif isinstance(height_map, list) and len(height_map) > 0:
                    print("普通数组")
                    height = len(height_map)
                    width = len(height_map[0]) if height > 0 else 0
            
            if width is None or height is None or width <= 0 or height <= 0:
                # 添加兼容性检查
                if hasattr(self.logger, 'log') and not hasattr(self.logger, 'info'):
                    # 使用log方法代替info
                    self.logger.log(f"从height_map推断地图尺寸: {w}x{h}")
                else:
                    # 使用标准info方法
                    self.logger.info(f"从height_map推断地图尺寸: {w}x{h}")
                raise ValueError("无法确定地图尺寸")
            print("转换成字典")        
            return {
                "width": width,
                "height": height,
                "height_map": height_map,
                "biome_map": map_data.get_layer("biome"),
                "vegetation": map_data.layers.get("vegetation", []),
                "buildings": map_data.layers.get("buildings", []),
                "rivers": map_data.get_layer("rivers"),
                "content_layout": getattr(map_data, "content_layout", {}),
                "caves": map_data.layers.get("caves", []),
                "map_params": getattr(map_data, "params", {}),
                "biome_data": getattr(map_data, "biome_data", None),
                "road_coords": map_data.layers.get("roads", []),
                "roads_map": (map_data.get_layer("roads_map"), map_data.get_layer("roads_types"))
            }
        print("已经是字典")
        return map_data  # 如果已经是字典，直接返回
    
    @abstractmethod
    def export(self, map_data: Dict, filename: str = None) -> str:
        """导出地图，返回生成的文件路径"""
        pass
    
    def _get_output_path(self, filename_base: str, extension: str) -> str:
        """构造输出文件路径"""
        if not filename_base:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"{self.config.base_filename}_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        return os.path.join(self.config.output_dir, f"{filename_base}.{extension}")
        
    def _generate_texture_image(self, map_data: Dict) -> Image.Image:
        """生成地图纹理贴图，使用高效的NumPy操作和渲染优先级"""       
        try:
            # 提取地图数据
            width, height = map_data.get("width", 0), map_data.get("height", 0)
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            caves = map_data.get("caves", [])
            road_coords = map_data.get("road_coords", [])
            content_layout = map_data.get("content_layout", {})
            
            # 道路网络处理
            roads_map = map_data.get("roads_map", (None, None))
            road_net, road_types = roads_map if isinstance(roads_map, tuple) and len(roads_map) == 2 else (None, None)
            
            # 验证地图尺寸
            if width <= 0 or height <= 0:
                print(f"无效的地图尺寸: {width}x{height}")
                return Image.new('RGB', (256, 256), (255, 0, 0))
            
            # 创建纹理图像
            texture_w, texture_h = self.config.texture_size
            
            # 计算纹理映射缩放因子
            scale_x = texture_w / width
            scale_y = texture_h / height
                
            # === 初始化布尔掩码用于高效索引 ===
            river_mask = np.zeros((height, width), dtype=bool)
            cave_mask = np.zeros((height, width), dtype=bool)
            veg_mask = np.zeros((height, width), dtype=bool)
            bld_mask = np.zeros((height, width), dtype=bool)
            story_mask = np.zeros((height, width), dtype=bool)
            creature_mask = np.zeros((height, width), dtype=bool)
            
            # 道路系统掩码
            settlement_roads_mask = np.zeros((height, width), dtype=bool)
            building_main_road = np.zeros((height, width), dtype=bool)  
            building_secondary = np.zeros((height, width), dtype=bool)
            building_paths = np.zeros((height, width), dtype=bool)
            
            # === 处理河流数据 ===
            if isinstance(rivers, np.ndarray):
                if rivers.shape == (height, width):
                    river_mask = rivers > 0
                elif rivers.shape[:2] == (height, width):
                    river_mask = np.any(rivers > 0, axis=-1) if rivers.ndim > 2 else rivers > 0
            elif isinstance(rivers, list) and len(rivers) > 0:
                for j in range(min(len(rivers), height)):
                    for i in range(min(len(rivers[j]), width)):
                        river_mask[j, i] = bool(rivers[j][i])
            
            # === 处理洞穴数据 ===
            if isinstance(caves, dict) and "caves" in caves:
                for point in caves.get("caves", []):
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        x, y = int(point["x"]), int(point["y"])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
            elif isinstance(caves, list):
                for item in caves:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        x, y = int(item[0]), int(item[1])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
                    elif isinstance(item, dict) and "x" in item and "y" in item:
                        x, y = int(item["x"]), int(item["y"])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
            
            # === 处理植被数据 ===
            for veg in vegetation:
                if isinstance(veg, (list, tuple)) and len(veg) >= 2:
                    x, y = int(veg[0]), int(veg[1])
                    if 0 <= x < width and 0 <= y < height:
                        veg_mask[y, x] = True
                elif isinstance(veg, dict) and "x" in veg and "y" in veg:
                    x, y = int(veg["x"]), int(veg["y"])
                    if 0 <= x < width and 0 <= y < height:
                        veg_mask[y, x] = True
            
            # === 处理建筑数据 ===
            for building in buildings:
                if isinstance(building, (list, tuple)) and len(building) >= 2:
                    x, y = int(building[0]), int(building[1])
                    if 0 <= x < width and 0 <= y < height:
                        bld_mask[y, x] = True
                elif isinstance(building, dict) and "x" in building and "y" in building:
                    x, y = int(building["x"]), int(building["y"])
                    if 0 <= x < width and 0 <= y < height:
                        bld_mask[y, x] = True
            
            # === 处理道路数据 ===
            for road in road_coords:
                if isinstance(road, (list, tuple)) and len(road) >= 2:
                    x, y = int(road[0]), int(road[1])
                    if 0 <= x < width and 0 <= y < height:
                        settlement_roads_mask[y, x] = True
                elif isinstance(road, dict) and "x" in road and "y" in road:
                    x, y = int(road["x"]), int(road["y"])
                    if 0 <= x < width and 0 <= y < height:
                        settlement_roads_mask[y, x] = True
            
            # === 处理道路网络 ===
            if road_net is not None and road_types is not None:
                if road_net.shape == (height, width) and road_types.shape == (height, width):
                    building_main_road = (road_net == True) & (road_types == 1)
                    building_secondary = (road_net == True) & (road_types == 2)
                    building_paths = (road_net == True) & (road_types == 3)
            
            # === 处理故事点和生物 ===
            if isinstance(content_layout, dict):
                if "story_events" in content_layout:
                    for event in content_layout["story_events"]:
                        if isinstance(event, dict) and "x" in event and "y" in event:
                            x, y = int(event["x"]), int(event["y"])
                            if 0 <= x < width and 0 <= y < height:
                                story_mask[y, x] = True
                
                if "creatures" in content_layout:
                    for creature in content_layout["creatures"]:
                        if isinstance(creature, dict) and "x" in creature and "y" in creature:
                            x, y = int(creature["x"]), int(creature["y"])
                            if 0 <= x < width and 0 <= y < height:
                                creature_mask[y, x] = True
            
            # === 生成生物群系颜色 ===
            biome_colors = np.ones((height, width, 3), dtype=np.float32) * 0.5  # 默认灰色
            
            for j in range(height):
                for i in range(width):
                    if j < len(biome_map) and i < len(biome_map[j]):
                        biome = biome_map[j][i]
                        # 根据生物群系类型处理颜色
                        if isinstance(biome, dict) and "color" in biome:
                            biome_colors[j, i] = biome["color"]
                        elif isinstance(biome, (int, np.integer)):
                            # 对于ID类型，可以在这里添加从ID到颜色的映射
                            # 使用取模运算生成不同的颜色
                            hue = (biome % 6) / 6.0
                            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
                            biome_colors[j, i] = [r, g, b]
            
            # === 创建高质量的纹理图像 ===
            # 初始化纹理数组
            texture_array = np.zeros((texture_h, texture_w, 3), dtype=np.uint8)
            
            # 定义地形特征颜色
            COLORS = {
                'river': (0, 128, 255),          # 河流蓝色
                'cave': (40, 40, 60),            # 洞穴深灰色
                'settlement_road': (230, 180, 80),  # 聚落道路橙色
                'main_road': (150, 50, 50),      # 主干道深红色
                'secondary': (180, 150, 130),    # 次干道土黄色
                'path': (150, 150, 150),         # 小路灰色
                'vegetation': (100, 180, 100),   # 植被绿色
                'building': (200, 100, 100),     # 建筑红棕色
                'story': (200, 80, 200),         # 故事点紫色
                'creature': (230, 230, 25)       # 生物黄色
            }
            
            # 创建坐标映射
            y_indices, x_indices = np.mgrid[0:texture_h, 0:texture_w]
            map_x = np.floor(x_indices / scale_x).astype(int)
            map_y = np.floor(y_indices / scale_y).astype(int)
            
            # 限制坐标范围
            valid_mask = (map_x < width) & (map_y < height) & (map_x >= 0) & (map_y >= 0)
            
            # 使用NumPy广播填充基础地形颜色
            for j in range(texture_h):
                for i in range(texture_w):
                    if valid_mask[j, i]:
                        map_j, map_i = map_y[j, i], map_x[j, i]
                        # 基础颜色是生物群系颜色
                        texture_array[j, i] = (biome_colors[map_j, map_i] * 255).astype(np.uint8)
                        
                        # 按照优先级应用特征颜色
                        if river_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['river']
                        
                        # 依次应用各种道路类型（优先级递增）
                        if building_paths[map_j, map_i]:
                            texture_array[j, i] = COLORS['path']
                        if building_secondary[map_j, map_i]:
                            texture_array[j, i] = COLORS['secondary']
                        if building_main_road[map_j, map_i]:
                            texture_array[j, i] = COLORS['main_road']
                        if settlement_roads_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['settlement_road']
                        
                        # 应用其他特征（优先级递增）
                        if veg_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['vegetation']
                        if cave_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['cave']
                        if bld_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['building']
                        if story_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['story']
                        if creature_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['creature']
            
            # 创建PIL图像
            img = Image.fromarray(texture_array, 'RGB')
            
            # 应用平滑滤镜增强视觉效果
            img = img.filter(Image.SMOOTH)
            
            return img
            
        except Exception as e:
            # 详细的错误日志
            import traceback
            print(f"生成纹理图像失败: {str(e)}")
            print(traceback.format_exc())
            # 返回一个简单的错误图像
            return Image.new('RGB', (256, 256), (255, 0, 0))
#############################################
#导出为obj文件
############################################# 
class ObjExporter(MapExporter):
    def __init__(self, config = None, logger=None):
        super().__init__(config, logger)
        self.logger=logger
    
    def export(self, map_data: Dict, filename: str = None) -> str:
        """导出为OBJ文件"""
        map_data_dict = self._prepare_map_data_dict(map_data)
        
        # 如果缺少宽度或高度，尝试从height_map推断
        if ("width" not in map_data_dict or "height" not in map_data_dict) and "height_map" in map_data_dict:
            height_map = map_data_dict["height_map"]
            # 修改这里的检查，避免对NumPy数组进行布尔测试
            if height_map is not None:
                if isinstance(height_map, np.ndarray):
                    h, w = height_map.shape
                else:
                    h = len(height_map)
                    w = len(height_map[0]) if h > 0 else 0
                map_data_dict["height"] = h
                map_data_dict["width"] = w
                self.logger.info(f"从height_map推断地图尺寸: {w}x{h}")
    
        # 标准化文件名
        if not filename:
            filename = self._get_output_path(None, "obj")
        
        # 确保必要字段存在
        if "width" not in map_data_dict or "height" not in map_data_dict or "height_map" not in map_data_dict:
            self.logger.error("导出数据缺少必要字段")
            return ""
        
        w = map_data_dict["width"]
        h = map_data_dict["height"]
        height_map = map_data_dict["height_map"]  
        
        # 生成并保存纹理
        texture_path = os.path.splitext(filename)[0] + "_texture.png"
        if self.config.generate_textures:
            try:
                texture_img = self._generate_texture_image(map_data_dict)  # 使用map_data_dict而不是map_data
                # 检查是否是错误图像
                if texture_img.size == (256, 256) and list(texture_img.getdata())[0] == (255, 0, 0):
                    print("警告：纹理生成返回了错误图像，将尝试使用简单的高度图替代")
                    # 尝试生成一个基于高度图的简单替代纹理
                    texture_img = self._generate_fallback_texture(map_data_dict)
                texture_img.save(texture_path)
            except Exception as e:
                print(f"保存纹理时出错: {e}")
                texture_path = None  # 在mtl文件中不引用纹理
        
        # 生成材质文件
        mtl_path = os.path.splitext(filename)[0] + ".mtl"
        with open(mtl_path, 'w') as mtl_file:
            mtl_file.write("newmtl MapTexture\n")
            mtl_file.write("Ka 1.0 1.0 1.0\n")  # 环境光
            mtl_file.write("Kd 1.0 1.0 1.0\n")  # 漫反射
            mtl_file.write("Ks 0.5 0.5 0.5\n")  # 高光
            mtl_file.write(f"map_Kd {os.path.basename(texture_path)}\n")
        
        # 计算LOD步长和尺寸
        lod_step = self.config.level_of_detail
        w_lod = (w - 1) // lod_step + 1
        h_lod = (h - 1) // lod_step + 1
        
        # 流式写入OBJ文件
        with open(filename, 'w') as obj_file:
            obj_file.write(f"mtllib {os.path.basename(mtl_path)}\n")
            obj_file.write("usemtl MapTexture\n")
            
            # 预计算顶点索引映射（内存高效模式下不存储）
            vertex_index = 1
            
            # 流式写入顶点数据
            for j in range(0, h, lod_step):
                for i in range(0, w, lod_step):
                    # 写入顶点坐标
                    y = height_map[j][i]
                    obj_file.write(f"v {i} {y} {j}\n")
                    
                    # 写入纹理坐标
                    u = i / (w - 1) if w > 1 else 0.0
                    v_val = 1.0 - (j / (h - 1)) if h > 1 else 0.0
                    obj_file.write(f"vt {u} {v_val}\n")
                    
                    # 计算并写入法线
                    if self.config.export_normals:
                        # 计算相邻点高度差
                        left = max(0, i - lod_step)
                        right = min(w-1, i + lod_step)
                        down = max(0, j - lod_step)
                        up = min(h-1, j + lod_step)
                        
                        # 计算梯度
                        dx = (height_map[j][right] - height_map[j][left]) / (right - left) if (right != left) else 0.0
                        dz = (height_map[up][i] - height_map[down][i]) / (up - down) if (up != down) else 0.0
                        
                        # 计算法线并归一化
                        normal = (-dx, 1.0, -dz)
                        length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                        if length == 0:
                            normal = (0.0, 1.0, 0.0)
                        else:
                            normal = (normal[0]/length, normal[1]/length, normal[2]/length)
                        obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                    
                    vertex_index += 1

            # 流式写入面数据
            for j in range(0, h - lod_step, lod_step):
                for i in range(0, w - lod_step, lod_step):
                    # 计算顶点索引
                    current_row = j // lod_step
                    current_col = i // lod_step
                    next_row = (j + lod_step) // lod_step
                    next_col = (i + lod_step) // lod_step
                    
                    v1 = current_row * w_lod + current_col + 1
                    v2 = current_row * w_lod + next_col + 1
                    v3 = next_row * w_lod + next_col + 1
                    v4 = next_row * w_lod + current_col + 1
                    
                    # 写入面（两个三角形）
                    if self.config.export_normals:
                        obj_file.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                        obj_file.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v4}/{v4}/{v4}\n")
                    else:
                        obj_file.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                        obj_file.write(f"f {v1}/{v1} {v3}/{v3} {v4}/{v4}\n")

        # 压缩处理
        if self.config.compress_output:
            with open(filename, 'rb') as f_in:
                with gzip.open(f"{filename}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filename)
            filename += ".gz"
        
        self.logger.info(f"成功导出OBJ文件到 {filename}")
        return filename

    def _generate_fallback_texture(self, map_data: Dict) -> Image.Image:
        """生成基于高度图的备用纹理"""
        try:
            w = map_data.get("width", 0)
            h = map_data.get("height", 0)
            height_map = map_data.get("height_map", [])
            
            if w <= 0 or h <= 0 or not height_map:
                return Image.new('RGB', (512, 512), (200, 200, 200))
            
            # 找出高度范围
            min_height = float('inf')
            max_height = float('-inf')
            for row in height_map:
                for height in row:
                    min_height = min(min_height, height)
                    max_height = max(max_height, height)
            
            height_range = max_height - min_height
            if height_range == 0:
                height_range = 1  # 避免除以零
            
            # 创建基于高度的简单纹理
            texture_w, texture_h = 512, 512
            img = Image.new('RGB', (texture_w, texture_h), (200, 200, 200))
            draw = ImageDraw.Draw(img)
            
            scale_x = texture_w / w
            scale_y = texture_h / h
            
            for j in range(h):
                for i in range(w):
                    # 根据高度生成颜色
                    normalized_height = (height_map[j][i] - min_height) / height_range
                    # 简单地图配色：低=蓝/绿，中=棕/绿，高=灰/白
                    if normalized_height < 0.3:  # 水域和低地
                        r, g, b = 100, 150 + int(100 * normalized_height), 200
                    elif normalized_height < 0.7:  # 平原和丘陵
                        r, g, b = 100 + int(155 * normalized_height), 150, 50
                    else:  # 山地
                        intensity = int(200 * normalized_height)
                        r, g, b = 150 + intensity//4, 150 + intensity//4, 150 + intensity//4
                    
                    x1, y1 = int(i * scale_x), int(j * scale_y)
                    x2, y2 = int((i+1) * scale_x), int((j+1) * scale_y)
                    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))
            
            return img
        except Exception as e:
            print(f"生成备用纹理失败: {e}")
            return Image.new('RGB', (512, 512), (200, 200, 200))
################################################
#unity导出器
################################################    
class UnityExporter(MapExporter, JSONExporterBase):
    """Unity专用导出器"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        # 确保两个父类都被正确初始化
        MapExporter.__init__(self, config,self.logger)
        JSONExporterBase.__init__(self, config,self.logger)
    
    def export(self, map_data: Dict, filename: str = None) -> str:
        # 将 MapData 对象转换为字典
        map_data = self._prepare_map_data_dict(map_data)
        normalized_data = self._prepare_export_data(map_data)
        filename = self._get_output_path("unity_map", "json")
        
        # 构建导出数据
        export_data = {
            "terrain": self._build_terrain_data(normalized_data),
            "entities": self._build_entities(normalized_data),
            "metadata": self._generate_metadata(normalized_data)
        }
        
        if not self._validate_data(export_data):
            return ""
        
        self._write_json_file(export_data, filename)
        return filename
    
    def _build_terrain_data(self, data: Dict) -> Dict:
        """构建地形数据"""
        return {
            "width": data["width"],
            "height": data["height"],
            "height_map": data["height_map"],
            "texture_map": self._generate_texture_path(),
            "biomes": self._process_biomes(data["biome_map"]),
            "lightmap_uvs": self.config.lightmap_uvs,
            "collision": self.config.export_collision
        }
    
    def _build_entities(self, data: Dict) -> Dict:
        """构建实体数据"""
        return {
            "vegetation": self._process_entities(data["vegetation"]),
            "buildings": self._process_entities(data["buildings"]),
            "rivers": self._process_rivers(data["rivers"]),
            "caves": self._process_caves(data["caves"]),
            "roads": self._process_roads(data["road_coords"])
        }
        
    def _get_output_path(self, filename_base: str, extension: str) -> str:
        """构造输出文件路径"""
        if not filename_base:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"{self.config.base_filename}_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        return os.path.join(self.config.output_dir, f"{filename_base}.{extension}")
    
    def _validate_data(self, data: Dict) -> bool:
        """验证Unity导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        if "terrain" not in data or "entities" not in data:
            self.logger.error("导出数据缺少必要字段: terrain 或 entities")
            return False
            
        # 验证地形数据
        terrain = data["terrain"]
        if not terrain or "width" not in terrain or "height" not in terrain or "height_map" not in terrain:
            self.logger.error("地形数据不完整")
            return False
            
        # 验证尺寸
        if terrain["width"] <= 0 or terrain["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {terrain['width']}x{terrain['height']}")
            return False
            
        return True
    
    def create_unity_package(self, map_data: Dict, filename: str = None) -> str:
        """创建Unity引擎专用包"""
        # 解包地图数据
        height_map = map_data.get("height_map", [])
        biome_map = map_data.get("biome_map", [])
        vegetation = map_data.get("vegetation", [])
        buildings = map_data.get("buildings", [])
        rivers = map_data.get("rivers", [])
        caves = map_data.get("caves", [])
        roads = map_data.get("road_coords", [])
        content_layout = map_data.get("content_layout", {})
        
        # 确保资源目录存在
        unity_dir = os.path.join(self.config.output_dir, "unity")
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator", exist_ok=True)
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Models", exist_ok=True)
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Textures", exist_ok=True)
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Materials", exist_ok=True)
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Prefabs", exist_ok=True)
        os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Scripts", exist_ok=True)
        
        # 创建高度图纹理
        heightmap_path = create_heightmap_texture(height_map)
        shutil.copy(heightmap_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
        
        # 创建法线贴图
        if self.config.export_normals:
            normal_map_path = create_normal_map(height_map)
            shutil.copy(normal_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
        
        # 创建材质混合贴图
        splat_map_path, biome_mapping = create_splat_map(biome_map)
        shutil.copy(splat_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
        
        # 创建地形网格
        terrain_mesh_path = create_enhanced_terrain_mesh(height_map)
        shutil.copy(terrain_mesh_path, f"{unity_dir}/Assets/TerrainGenerator/Models/")
        
        # 创建材质文件
        sea_level = 15.0  # 默认海平面高度
        materials_path = create_material_files(biome_mapping, sea_level)
        shutil.copy(materials_path, f"{unity_dir}/Assets/TerrainGenerator/Materials/")
        
        # 放置3D模型
        models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout)
        with open(f"{unity_dir}/Assets/TerrainGenerator/model_placements.json", "w") as f:
            json.dump(models, f, indent=2)
        
        # 创建Unity地形设置
        with open(f"{unity_dir}/Assets/TerrainGenerator/terrain_settings.json", "w") as f:
            json.dump({
                "width": len(height_map[0]),
                "height": len(height_map),
                "heightScale": 1.0,
                "heightmapResolution": len(height_map),
                "detailResolution": 1024,
                "controlTextureResolution": 1024,
                "baseTextureResolution": 1024,
                "seaLevel": sea_level,
                "unityVersion": self.config.unity_export_version,
                "lightmapUVs": self.config.lightmap_uvs
            }, f, indent=2)
        
        # 创建Unity导入器脚本
        with open(f"{unity_dir}/Assets/TerrainGenerator/Scripts/TerrainImporter.cs", "w") as f:
            f.write("""
    using UnityEngine;
    using System.Collections.Generic;
    using System.IO;
    using UnityEditor;

    #if UNITY_EDITOR
    public class TerrainImporter : EditorWindow {
        [MenuItem("Tools/Import Generated Terrain")]
        static void Init() {
            TerrainImporter window = GetWindow<TerrainImporter>();
            window.Show();
        }

        void OnGUI() {
            if(GUILayout.Button("Import Terrain and Assets")) {
                ImportAll();
            }
        }

        void ImportAll() {
            // 读取配置文件
            string settingsPath = "Assets/TerrainGenerator/terrain_settings.json";
            string jsonText = File.ReadAllText(settingsPath);
            var settings = JsonUtility.FromJson<TerrainSettings>(jsonText);
            
            // 创建地形
            TerrainData terrainData = new TerrainData();
            terrainData.heightmapResolution = settings.heightmapResolution;
            terrainData.size = new Vector3(settings.width, settings.heightScale, settings.height);
            
            // 加载高度图
            Texture2D heightmap = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/TerrainGenerator/Textures/heightmap.png");
            float[,] heights = new float[settings.heightmapResolution, settings.heightmapResolution];
            
            // 设置高度图
            for (int y = 0; y < settings.heightmapResolution; y++) {
                for (int x = 0; x < settings.heightmapResolution; x++) {
                    float normX = (float)x / settings.heightmapResolution;
                    float normY = (float)y / settings.heightmapResolution;
                    Color pixelColor = heightmap.GetPixel(
                        Mathf.FloorToInt(normX * heightmap.width),
                        Mathf.FloorToInt(normY * heightmap.height)
                    );
                    heights[y, x] = pixelColor.grayscale;
                }
            }
            terrainData.SetHeights(0, 0, heights);
            
            // 保存地形数据
            AssetDatabase.CreateAsset(terrainData, "Assets/TerrainGenerator/TerrainData.asset");
            
            // 创建地形GameObject
            GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
            terrainGO.name = "GeneratedTerrain";
            
            // 放置模型
            ImportModels();
            
            Debug.Log("Terrain import complete!");
        }
        
        void ImportModels() {
            string modelDataPath = "Assets/TerrainGenerator/model_placements.json";
            string jsonText = File.ReadAllText(modelDataPath);
            ModelPlacementData placementData = JsonUtility.FromJson<ModelPlacementData>(jsonText);
            
            foreach (var model in placementData.models) {
                // 加载模型预制体
                string modelPath = "Assets/TerrainGenerator/Models/" + Path.GetFileName(model.model_path);
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
                
                if (prefab != null) {
                    GameObject instance = Instantiate(prefab);
                    instance.transform.position = new Vector3(model.position[0], model.position[2], model.position[1]);
                    instance.transform.eulerAngles = new Vector3(model.rotation[0], model.rotation[1], model.rotation[2]);
                    instance.transform.localScale = Vector3.one * model.scale;
                    instance.name = model.type;
                }
            }
        }
    }

    [System.Serializable]
    public class TerrainSettings {
        public int width;
        public int height;
        public float heightScale;
        public int heightmapResolution;
        public int detailResolution;
        public int controlTextureResolution;
        public int baseTextureResolution;
        public float seaLevel;
    }

    [System.Serializable]
    public class ModelPlacement {
        public string type;
        public float[] position;
        public float[] rotation;
        public float scale;
        public string model_path;
    }

    [System.Serializable]
    public class ModelPlacementData {
        public List<ModelPlacement> models;
    }
    #endif
            """)
        
        # 创建README.md
        with open(f"{unity_dir}/README.md", "w") as f:
            f.write(f"""
    # Generated Unity Terrain Package

    This package contains a procedurally generated terrain with models, materials and game content.

    ## Import Instructions

    1. Create a new Unity project or open an existing one (Version {self.config.unity_export_version} recommended)
    2. Copy the contents of this directory into your Assets folder
    3. Open the Unity Editor
    4. Go to Tools > Import Generated Terrain
    5. Click "Import Terrain and Assets"
    6. The terrain will be generated with all models placed as specified

    ## Content

    - Complete 3D terrain with height data
    - Materials and textures
    - Placed 3D models for vegetation, buildings, and other features
    - Game content data including creature attributes and story events
    - Rivers, roads, and other path systems
            """)
        
        # 返回Unity目录路径
        return unity_dir