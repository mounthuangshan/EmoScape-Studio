#from __future__ import annotations
#标准库
import json
import random
import time
#import logging

#数据处理与科学计算

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发

#其他工具
from openai import OpenAI
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable

#项目文件
from pygamemap import *


#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')# 时间戳，日志级别，信息
#logger = logging.getLogger(__name__)

class LLMIntegration:
    """LLM集成类，处理所有与LLM相关的交互"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def query_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """发送查询到LLM，带重试机制"""
        for attempt in range(max_retries):
            try:
                self.logger.log(f"向LLM发送查询 (尝试 {attempt+1}/{max_retries})")
                response = self._send_query(prompt)
                return response
            except Exception as e:
                self.logger.error(f"LLM查询失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 延迟重试
                else:
                    return f"LLM查询失败: {str(e)}"
    
    def _send_query(self, prompt: str) -> str:
        """发送查询到LLM (内部函数)"""
        try:
            # 调用query_llm函数
            return self.query_llm(prompt)
        except ImportError:
            # 模块导入失败时的回退行为
            self.logger.warning("无法导入query_llm函数，使用模拟响应")
            return "LLM集成不可用。请检查API密钥配置。"
    
    def get_parameter_suggestions(self, current_params: MapParameters) -> Dict[str, Any]:
        """获取LLM对参数的建议"""
        prompt = f"""
        当前地图生成参数如下，请根据游戏设计最佳实践提供优化建议:
        Achievement: {current_params.achievement}
        Exploration: {current_params.exploration}
        Social: {current_params.social}
        Combat: {current_params.combat}
        Map Width: {current_params.map_width}
        Map Height: {current_params.map_height}
        Vegetation Coverage: {current_params.vegetation_coverage}
        City Count: {current_params.city_count}
        
        根据这些参数，请提出能改善玩家体验的调整建议。
        请以JSON格式返回建议的参数值，仅包含需要调整的参数。格式示例:
        {{"achievement": 0.6, "exploration": 0.7}}
        """
        
        response = self.query_with_retry(prompt)
        
        try:
            # 尝试从响应中提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            self.logger.error("无法解析LLM响应为JSON")
            return {}

    def query_llm(self,prompt):
        if OpenAI.api_key is None:
            return "无LLM可用。"
        client = OpenAI(base_url  = "https://api.moonshot.cn/v1")
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.3
        );
        return completion.choices[0].message.content

    ################
    #布局微调和故事生成
    ################
    def process_llm_tasks(self, content_layout, biome_map, preferences):
        """综合处理LLM任务：1. 布局微调 2. 故事生成"""
        # 获取地图尺寸
        if biome_map.ndim == 2:
            map_height, map_width = biome_map.shape
        else:
            biome_map = biome_map.reshape(-1, biome_map.size)
            map_height, map_width = biome_map.shape

        # 任务1: 布局微调
        try:
            layout_response = self.query_llm(
                f"根据玩家偏好【{preferences}】对地图布局进行微调，特别是特殊物品的放置。"
            )
            print("布局LLM响应:", layout_response)
            content_layout["objects"].append({"type": "npc", "x": 10, "y": 2})
        except Exception as e:
            print(f"布局生成错误: {e}")
            layout_response = "布局生成失败"

        # 任务2: 故事生成
        try:
            story_prompt = f"""
            根据以下玩家偏好和地图特征，为本地图生成故事背景，并提出3个特定坐标的故事事件：
            偏好：{preferences}
            地图尺寸：{map_width}x{map_height}
            请在回答中包含三个(x,y)坐标的事件点及对应情节描述。
            坐标格式示例：坐标(5,10)处有一个神秘洞穴...
            必须返回3个完整的坐标点和描述。
            """
            story_response = self.query_llm(story_prompt)
            print("故事LLM响应:", story_response)

            # 解析 LLM 响应中的坐标
            import re
            story_events = []
            lines = [line.strip() for line in story_response.split('\n') if line.strip()]

            for line in lines:
                coords = re.search(r'\((\d+),(\d+)\)', line)
                if coords:
                    try:
                        x = int(coords.group(1))
                        y = int(coords.group(2))
                        x = min(max(0, x), map_width - 1)
                        y = min(max(0, y), map_height - 1)
                        story_events.append({
                            "x": x,
                            "y": y,
                            "type": "story_event",
                            "description": line.strip()
                        })
                    except ValueError:
                        print(f"无法将坐标转换为整数: {coords.groups()}")

            # 确保至少有3个故事事件
            required_events = 3
            if len(story_events) < required_events:
                print(f"从LLM响应中提取了{len(story_events)}个坐标，少于所需的最少{required_events}个，使用随机坐标补充")
                default_descriptions = [
                    "神秘洞穴: 一个隐藏的洞穴等待探索",
                    "古代遗迹: 被时间遗忘的古老文明痕迹",
                    "稀有资源: 一处蕴藏珍贵矿物的地点"
                ]
                for i in range(required_events - len(story_events)):
                    x = random.randint(0, map_width - 1)
                    y = random.randint(0, map_height - 1)
                    description = default_descriptions[i % len(default_descriptions)]
                    story_events.append({
                        "x": int(x),
                        "y": int(y),
                        "type": "story_event",
                        "description": description
                    })
                    print(f"添加随机故事事件: x={x}, y={y}, 描述={description}")

        except Exception as e:
            print(f"故事生成错误: {e}")
            story_events = [
                {"x": int(random.randint(0, map_width - 1)), "y": int(random.randint(0, map_height - 1)), "type": "story_event", "description": f"默认故事点 {i+1}"}
                for i in range(3)
            ]
            story_response = "故事生成失败"

        # 验证所有事件
        valid_story_events = []
        for event in story_events:
            if isinstance(event, dict) and "x" in event and "y" in event and "description" in event:
                event["x"] = int(event["x"])
                event["y"] = int(event["y"])
                valid_story_events.append(event)
                print(f"验证通过的事件: x={event['x']}, y={event['y']}, 描述={event['description'][:30]}...")
            else:
                print(f"丢弃无效事件: {event}")

        # 更新 content_layout
        content_layout.update({
            "story_events": valid_story_events,
            "story_overview": story_response,
            "llm_debug": {
                "layout_prompt": layout_response,
                "story_prompt": story_response
            }
        })
        return content_layout
    
    def generate_story_expansion(self, story_events, biome_map, preferences):
        """扩展故事事件点，生成完整情节的游戏剧情
        
        Args:
            story_events: 故事事件列表
            biome_map: 生物群系地图
            preferences: 用户偏好设置
            
        Returns:
            包含扩展的故事内容的字典
        """
        if not story_events:
            self.logger.log("没有故事事件可扩充", "WARNING")
            return {"overall_story": "", "expanded_stories": []}
        
        # 获取地图尺寸
        map_height, map_width = 0, 0
        if biome_map is not None:
            if hasattr(biome_map, 'shape'):
                map_height, map_width = biome_map.shape
            else:
                self.logger.log("无法确定地图尺寸，使用默认值", "WARNING")
        
        # 构建整体故事线提示
        overall_prompt = f"""
        你是一位专业游戏剧情设计师。请基于以下故事事件点，创建一个连贯的游戏剧情故事线。
        地图尺寸: {map_width}x{map_height}
        玩家偏好: {preferences}
        
        故事事件点:
        """
        
        for i, event in enumerate(story_events):
            if not isinstance(event, dict):
                continue
            overall_prompt += f"{i+1}. 位置({event.get('x', '?')},{event.get('y', '?')}): {event.get('description', '未知事件')}\n"
        
        overall_prompt += """
        请创建一个连贯的游戏主线剧情，包含以下内容:
        1. 故事背景和世界设定
        2. 主要角色介绍（玩家角色和关键NPC）
        3. 主线任务流程
        4. 如何将以上事件点融入到主线剧情中
        5. 分支故事线可能性
        6. 世界历史和传说

        请提供结构化的、游戏玩家能直接体验的叙事内容，不要使用游戏设计文档格式。
        回复格式请使用Markdown，以便更好地组织和显示内容。
        """
        
        # 调用LLM生成整体剧情
        self.logger.log("正在生成游戏整体剧情故事线...")
        overall_story = self.query_with_retry(overall_prompt)
        
        # 然后为每个事件点生成详细的任务和对话
        expanded_stories = []
        for i, event in enumerate(story_events):
            if not isinstance(event, dict):
                continue
            time.sleep(1)    
            x = event.get('x', 0)
            y = event.get('y', 0)
            description = event.get('description', '未知事件')
            
            event_prompt = f"""
            基于以下故事事件点，详细设计一个游戏任务/事件:
            
            事件描述: {description}
            事件位置: ({x},{y})
            
            请提供以下内容:
            1. 任务名称
            2. 任务类型（主线/支线/隐藏）
            3. 任务前置条件
            4. 详细任务描述
            5. NPC对话脚本（包括对话选项和分支）
            6. 任务目标和奖励
            7. 与主线故事的联系
            
            请提供游戏中可以直接使用的内容，使用Markdown格式。
            """
            
            self.logger.log(f"正在为事件点{i+1}生成详细任务...")
            event_story = self.query_with_retry(event_prompt)
            
            expanded_stories.append({
                "original_event": event,
                "expanded_content": event_story
            })
        
        # 返回整合的剧情内容
        return {
            "overall_story": overall_story,
            "expanded_stories": expanded_stories
        }