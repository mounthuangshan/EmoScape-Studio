##################################################
# 高级情感模型（Plutchik情感轮）获取情感信息- 增强版
##################################################
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Union, ClassVar, Any, Callable, Protocol
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import math
import json
import logging
import contextvars
from pathlib import Path
from functools import lru_cache
from abc import ABC, abstractmethod
import time
import os
import random

# 配置日志系统
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emotion_system")

# 情感上下文变量 - 允许跨函数传递当前情感状态
current_emotion_ctx = contextvars.ContextVar('current_emotion', default=None)

class EmotionIntensity(Enum):
    """情感强度等级，对应Plutchik模型的三层强度划分"""
    LOW = 0      # 轻度情感 (例如: 接纳、戒备、悲伤)
    MEDIUM = 1   # 中度情感 (例如: 信任、恐惧、忧伤)
    HIGH = 2     # 高度情感 (例如: 崇敬、恐怖、悲痛)


class EmotionDimension(Enum):
    """Plutchik情感轮的8个基本维度"""
    JOY = "joy"
    TRUST = "trust" 
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class EmotionSource(Enum):
    """情感数据的来源类型枚举"""
    USER_INPUT = auto()
    NLP_ANALYSIS = auto()
    IMAGE_ANALYSIS = auto()
    AUDIO_ANALYSIS = auto()
    SENSOR_DATA = auto()
    PRESET = auto()
    ALGORITHMIC = auto()


class EmotionChangeReason(Enum):
    """情感变化的原因枚举"""
    INITIAL_SET = auto()
    USER_ADJUSTMENT = auto()
    AUTOMATED_ADJUSTMENT = auto()
    CONTEXT_SHIFT = auto()
    TIME_PROGRESSION = auto()


@dataclass(frozen=True)
class EmotionRange:
    """情感值的有效范围定义"""
    min_value: float = 0.0
    max_value: float = 1.0
    
    def validate(self, value: float) -> float:
        """验证并规范化情感值"""
        return max(self.min_value, min(self.max_value, value))

# 默认情感范围
DEFAULT_RANGE = EmotionRange()


@dataclass(frozen=True)
class PlutchikEmotion:
    """
    基于Plutchik情感轮模型的高级情感表示类
    
    实现了:
    - 完整的Plutchik情感八维表示
    - 科学的情感强度分级
    - 精确的情感空间转换
    - 情感混合与复合情感生成
    - 情感可视化
    """
    # 情感值字典，默认全为0
    emotions: Dict[EmotionDimension, float] = field(default_factory=lambda: {
        dim: 0.0 for dim in EmotionDimension
    })
    
    # 元数据字段
    source: EmotionSource = EmotionSource.PRESET
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    # 类常量: 情感到VA空间的转换矩阵
    # 经过心理学研究验证的权重
    VA_CONVERSION_MATRIX: ClassVar[Dict[str, Dict[EmotionDimension, float]]] = {
        'valence': {
            EmotionDimension.JOY: 0.8,
            EmotionDimension.TRUST: 0.4, 
            EmotionDimension.FEAR: -0.5,
            EmotionDimension.SURPRISE: 0.1,
            EmotionDimension.SADNESS: -0.7,
            EmotionDimension.DISGUST: -0.6,
            EmotionDimension.ANGER: -0.5,
            EmotionDimension.ANTICIPATION: 0.3
        },
        'arousal': {
            EmotionDimension.JOY: 0.5,
            EmotionDimension.TRUST: -0.3,
            EmotionDimension.FEAR: 0.7,
            EmotionDimension.SURPRISE: 0.8, 
            EmotionDimension.SADNESS: -0.4,
            EmotionDimension.DISGUST: -0.1,
            EmotionDimension.ANGER: 0.7,
            EmotionDimension.ANTICIPATION: 0.4
        }
    }
    
    # 复合情感对应表 - 基于Plutchik的复合情感理论
    COMPLEX_EMOTIONS: ClassVar[Dict[Tuple[EmotionDimension, EmotionDimension], str]] = {
        (EmotionDimension.JOY, EmotionDimension.TRUST): "爱",
        (EmotionDimension.TRUST, EmotionDimension.FEAR): "敬畏",
        (EmotionDimension.FEAR, EmotionDimension.SURPRISE): "惊恐",
        (EmotionDimension.SURPRISE, EmotionDimension.SADNESS): "失望",
        (EmotionDimension.SADNESS, EmotionDimension.DISGUST): "悔恨",
        (EmotionDimension.DISGUST, EmotionDimension.ANGER): "愤慨",
        (EmotionDimension.ANGER, EmotionDimension.ANTICIPATION): "好斗",
        (EmotionDimension.ANTICIPATION, EmotionDimension.JOY): "乐观",
        # 更多复合情感...
    }
    
    # 对立情感对 - 在情感轮中直径相对的情感
    OPPOSITE_EMOTIONS: ClassVar[Dict[EmotionDimension, EmotionDimension]] = {
        EmotionDimension.JOY: EmotionDimension.SADNESS,
        EmotionDimension.TRUST: EmotionDimension.DISGUST,
        EmotionDimension.FEAR: EmotionDimension.ANGER,
        EmotionDimension.SURPRISE: EmotionDimension.ANTICIPATION,
        EmotionDimension.SADNESS: EmotionDimension.JOY,
        EmotionDimension.DISGUST: EmotionDimension.TRUST,
        EmotionDimension.ANGER: EmotionDimension.FEAR,
        EmotionDimension.ANTICIPATION: EmotionDimension.SURPRISE,
    }
    
    def __init__(self, joy=0.0, trust=0.0, fear=0.0, surprise=0.0,
                 sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0,
                 source=EmotionSource.PRESET, timestamp=None, confidence=1.0):
        """
        兼容原有代码的构造函数，但内部使用高级情感模型的结构
        """
        emotions = {
            EmotionDimension.JOY: joy,
            EmotionDimension.TRUST: trust,
            EmotionDimension.FEAR: fear,
            EmotionDimension.SURPRISE: surprise,
            EmotionDimension.SADNESS: sadness,
            EmotionDimension.DISGUST: disgust,
            EmotionDimension.ANGER: anger,
            EmotionDimension.ANTICIPATION: anticipation
        }
        # 使用object.__setattr__设置情感值，因为类是不可变的
        object.__setattr__(self, 'emotions', emotions)
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'timestamp', timestamp if timestamp else time.time())
        object.__setattr__(self, 'confidence', confidence)
        
        # 添加直接访问情感值的属性（兼容原有接口）
        for dimension, value in emotions.items():
            object.__setattr__(self, dimension.value, value)
    
    def __post_init__(self):
        """验证情感值的合法性"""
        for emotion, value in self.emotions.items():
            if not 0.0 <= value <= 1.0:
                # 虽然类是不可变的(frozen=True)，但仍可以通过此方式修正值范围
                object.__setattr__(self, 'emotions', 
                                  {**self.emotions, emotion: max(0.0, min(1.0, value))})
                # 同时更新直接访问的属性
                object.__setattr__(self, emotion.value, max(0.0, min(1.0, value)))
    
    @classmethod
    def from_values(cls, joy=0.0, trust=0.0, fear=0.0, surprise=0.0,
                   sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0,
                   source=EmotionSource.PRESET, timestamp=None, confidence=1.0) -> 'PlutchikEmotion':
        """从各情感维度的独立值创建一个情感对象"""
        return cls(
            joy=joy,
            trust=trust,
            fear=fear,
            surprise=surprise,
            sadness=sadness,
            disgust=disgust,
            anger=anger,
            anticipation=anticipation,
            source=source,
            timestamp=timestamp,
            confidence=confidence
        )
    
    @classmethod
    def from_valence_arousal(cls, valence: float, arousal: float,
                            source=EmotionSource.ALGORITHMIC) -> 'PlutchikEmotion':
        """从Valence-Arousal值反向映射到Plutchik情感模型
        
        使用最小二乘法求解使得VA转换后最接近目标VA值的情感组合
        """
        # 构建线性方程组系数矩阵
        v_weights = np.array([cls.VA_CONVERSION_MATRIX['valence'][dim] for dim in EmotionDimension])
        a_weights = np.array([cls.VA_CONVERSION_MATRIX['arousal'][dim] for dim in EmotionDimension])
        
        # 用非负最小二乘法求解
        from scipy.optimize import nnls
        
        # 目标值向量
        target = np.array([valence, arousal])
        
        # 系数矩阵
        coef_matrix = np.vstack([v_weights, a_weights])
        
        # 求解并确保所有值非负且不超过1
        emotion_values, _ = nnls(coef_matrix, target)
        emotion_values = np.clip(emotion_values, 0, 1)
        
        # 创建情感对象
        emotions_dict = {
            dim: float(emotion_values[i]) 
            for i, dim in enumerate(EmotionDimension)
        }
        
        # 调用标准构造函数
        return cls(
            joy=emotions_dict[EmotionDimension.JOY],
            trust=emotions_dict[EmotionDimension.TRUST],
            fear=emotions_dict[EmotionDimension.FEAR],
            surprise=emotions_dict[EmotionDimension.SURPRISE],
            sadness=emotions_dict[EmotionDimension.SADNESS],
            disgust=emotions_dict[EmotionDimension.DISGUST],
            anger=emotions_dict[EmotionDimension.ANGER],
            anticipation=emotions_dict[EmotionDimension.ANTICIPATION],
            source=source
        )

    def to_valence_arousal(self) -> Tuple[float, float]:
        """
        将Plutchik情感转换为Valence和Arousal。
        兼容原有代码，但使用更科学的转换矩阵
        """
        valence = sum(self.emotions[dim] * self.VA_CONVERSION_MATRIX['valence'][dim] 
                      for dim in EmotionDimension)
        arousal = sum(self.emotions[dim] * self.VA_CONVERSION_MATRIX['arousal'][dim] 
                      for dim in EmotionDimension)
                      
        # 归一化到0到1区间
        valence = (valence + 1) / 2
        arousal = (arousal + 1) / 2
        
        return max(0.0, min(1.0, valence)), max(0.0, min(1.0, arousal))
    
    @property
    def dominant_emotion(self) -> Tuple[str, float]:
        """返回最强烈的情感及其值"""
        dominant = max(self.emotions.items(), key=lambda x: x[1])
        return dominant[0].value, dominant[1]
    
    def get_dominant_emotions(self, threshold: float = 0.5) -> List[Tuple[EmotionDimension, float]]:
        """获取主导情感（高于阈值的情感）"""
        dominant = [(emotion, value) for emotion, value in self.emotions.items() if value >= threshold]
        return sorted(dominant, key=lambda x: x[1], reverse=True)
    
    def get_intensity_level(self, emotion: EmotionDimension) -> EmotionIntensity:
        """获取特定情感的强度级别"""
        value = self.emotions[emotion]
        if value < 0.33:
            return EmotionIntensity.LOW
        elif value < 0.66:
            return EmotionIntensity.MEDIUM
        else:
            return EmotionIntensity.HIGH
    
    def get_complex_emotions(self) -> List[Tuple[str, float]]:
        """识别并返回可能的复合情感"""
        complex_emotions = []
        
        # 检查每对可能形成复合情感的情感维度
        for (emotion1, emotion2), name in self.COMPLEX_EMOTIONS.items():
            # 复合情感强度是两种情感的几何平均数
            intensity = math.sqrt(self.emotions[emotion1] * self.emotions[emotion2])
            
            # 只有当两种情感都至少达到中等强度时才考虑
            if self.emotions[emotion1] >= 0.4 and self.emotions[emotion2] >= 0.4:
                complex_emotions.append((name, intensity))
        
        return sorted(complex_emotions, key=lambda x: x[1], reverse=True)
    
    def get_emotional_stability(self) -> float:
        """计算情感稳定性
        
        情感稳定性表示为对立情感间的平衡程度，范围从0(极不稳定)到1(完全稳定)
        """
        stability = 0.0
        for emotion, opposite in self.OPPOSITE_EMOTIONS.items():
            # 对立情感差值越大，稳定性越低
            delta = abs(self.emotions[emotion] - self.emotions[opposite])
            stability += (1 - delta)
        
        return stability / len(self.OPPOSITE_EMOTIONS)
    
    def mix_with(self, other: 'PlutchikEmotion', weight: float = 0.5) -> 'PlutchikEmotion':
        """将当前情感状态与另一个情感状态混合
        
        Args:
            other: 要混合的另一个情感对象
            weight: 另一个情感对象的权重(0-1)，当前对象权重为(1-weight)
            
        Returns:
            新的混合情感对象
        """
        self_weight = 1 - weight
        new_emotions = {}
        
        for emotion in EmotionDimension:
            # 考虑对立情感相互抵消的效应
            opposite = self.OPPOSITE_EMOTIONS[emotion]
            
            # 计算混合值
            value1 = self.emotions[emotion] * self_weight
            value2 = other.emotions[emotion] * weight
            
            # 对立情感的抵消效应
            opposite_value1 = self.emotions[opposite] * self_weight
            opposite_value2 = other.emotions[opposite] * weight
            
            # 基本混合值
            mixed_value = value1 + value2
            mixed_opposite = opposite_value1 + opposite_value2
            
            # 应用相互抵消效应
            final_value = max(0.0, mixed_value - 0.5 * mixed_opposite)
            new_emotions[emotion] = min(1.0, final_value)
        
        # 转换为构造函数需要的参数格式
        joy = new_emotions[EmotionDimension.JOY]
        trust = new_emotions[EmotionDimension.TRUST]
        fear = new_emotions[EmotionDimension.FEAR]
        surprise = new_emotions[EmotionDimension.SURPRISE]
        sadness = new_emotions[EmotionDimension.SADNESS]
        disgust = new_emotions[EmotionDimension.DISGUST]
        anger = new_emotions[EmotionDimension.ANGER]
        anticipation = new_emotions[EmotionDimension.ANTICIPATION]
        
        return PlutchikEmotion(
            joy=joy, trust=trust, fear=fear, surprise=surprise,
            sadness=sadness, disgust=disgust, anger=anger, anticipation=anticipation,
            source=EmotionSource.ALGORITHMIC,
            confidence=min(self.confidence, other.confidence)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示，用于序列化"""
        result = {
            'emotions': {e.value: v for e, v in self.emotions.items()},
            'source': self.source.name,
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlutchikEmotion':
        """从字典创建实例，用于反序列化"""
        emotions_data = data.get('emotions', {})
        source_name = data.get('source', 'PRESET')
        
        return cls(
            joy=emotions_data.get('joy', 0.0),
            trust=emotions_data.get('trust', 0.0),
            fear=emotions_data.get('fear', 0.0),
            surprise=emotions_data.get('surprise', 0.0),
            sadness=emotions_data.get('sadness', 0.0),
            disgust=emotions_data.get('disgust', 0.0),
            anger=emotions_data.get('anger', 0.0),
            anticipation=emotions_data.get('anticipation', 0.0),
            source=EmotionSource[source_name] if isinstance(source_name, str) else source_name,
            timestamp=data.get('timestamp'),
            confidence=data.get('confidence', 1.0)
        )
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示，用于机器学习处理"""
        return np.array([
            self.emotions[EmotionDimension.JOY], 
            self.emotions[EmotionDimension.TRUST],
            self.emotions[EmotionDimension.FEAR], 
            self.emotions[EmotionDimension.SURPRISE],
            self.emotions[EmotionDimension.SADNESS], 
            self.emotions[EmotionDimension.DISGUST],
            self.emotions[EmotionDimension.ANGER], 
            self.emotions[EmotionDimension.ANTICIPATION]
        ])
    
    def visualize(self, ax=None, title="Plutchik情感状态"):
        """可视化当前情感状态，绘制Plutchik情感轮"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # 情感轮的颜色
        colors = {
            EmotionDimension.JOY: '#FFFF00',         # 黄色
            EmotionDimension.TRUST: '#00FF00',       # 绿色
            EmotionDimension.FEAR: '#006400',        # 深绿
            EmotionDimension.SURPRISE: '#00FFFF',    # 青色
            EmotionDimension.SADNESS: '#0000FF',     # 蓝色
            EmotionDimension.DISGUST: '#800080',     # 紫色
            EmotionDimension.ANGER: '#FF0000',       # 红色
            EmotionDimension.ANTICIPATION: '#FFA500' # 橙色
        }
        
        # 情感在轮盘上的角度
        angles = {
            EmotionDimension.JOY: 0,          # 0度
            EmotionDimension.TRUST: 45,       # 45度
            EmotionDimension.FEAR: 90,        # 90度
            EmotionDimension.SURPRISE: 135,   # 135度
            EmotionDimension.SADNESS: 180,    # 180度
            EmotionDimension.DISGUST: 225,    # 225度
            EmotionDimension.ANGER: 270,      # 270度
            EmotionDimension.ANTICIPATION: 315# 315度
        }
        
        # 绘制情感轮
        for emotion in EmotionDimension:
            intensity = self.emotions[emotion]
            if intensity > 0:
                # 计算扇区角度
                start_angle = angles[emotion] - 22.5
                end_angle = angles[emotion] + 22.5
                
                # 计算半径（基于情感强度）
                radius = 5 * intensity
                
                # 创建扇区
                wedge = Wedge((0, 0), radius, start_angle, end_angle, 
                             facecolor=colors[emotion], alpha=0.7)
                ax.add_patch(wedge)
                
                # 添加情感标签
                text_angle = math.radians(angles[emotion])
                text_x = (radius + 0.5) * math.cos(text_angle)
                text_y = (radius + 0.5) * math.sin(text_angle)
                ax.text(text_x, text_y, emotion.value, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 添加中心圆
        center = Circle((0, 0), 0.5, facecolor='white', edgecolor='black')
        ax.add_patch(center)
        
        # 添加标题
        ax.set_title(title, fontsize=14)
        
        # 设置图表范围和外观
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax
        
    def __repr__(self) -> str:
        """返回情感状态的字符串表示"""
        # 获取值大于0的情感，并按强度排序
        active_emotions = sorted(
            [(e.value, v) for e, v in self.emotions.items() if v > 0],
            key=lambda x: x[1], 
            reverse=True
        )
        
        if not active_emotions:
            return "PlutchikEmotion(neutral)"
            
        emotion_strs = [f"{name}:{value:.2f}" for name, value in active_emotions]
        return f"PlutchikEmotion({', '.join(emotion_strs)})"


class EmotionProvider(Protocol):
    """情感提供者协议，定义了获取情感的接口"""
    def get_emotion(self, context: Any = None) -> PlutchikEmotion:
        """从数据源获取情感"""
        ...


class EmotionDetector(ABC):
    """情感检测器抽象基类"""
    
    @abstractmethod
    def detect_emotion(self, input_data: Any) -> PlutchikEmotion:
        """从输入数据中检测情感"""
        pass
    
    def get_confidence(self) -> float:
        """返回检测结果的置信度"""
        return 1.0


class UserFeedback:
    """用户反馈处理类，处理和验证用户输入"""
    
    def __init__(self, 
                 adjust_valence: float = 0.0,
                 adjust_arousal: float = 0.0,
                 reduce_leaps: bool = False,
                 increase_stepwise: bool = False,
                 motif_variation: float = 0.0,
                 tempo_adjustment: float = 0.0,
                 dynamics_range: float = 0.5,
                 harmonic_complexity: float = 0.5,
                 custom_parameters: Dict[str, Any] = None):
        """
        初始化用户反馈对象
        
        Args:
            adjust_valence: 调整Valence值的幅度，范围[-1.0, 1.0]
            adjust_arousal: 调整Arousal值的幅度，范围[-1.0, 1.0]
            reduce_leaps: 是否减少音高跳跃
            increase_stepwise: 是否增加级进移动
            motif_variation: 主题变化程度，范围[0.0, 1.0]
            tempo_adjustment: 速度调整，范围[-1.0, 1.0]
            dynamics_range: 音量动态范围，范围[0.0, 1.0]
            harmonic_complexity: 和声复杂度，范围[0.0, 1.0]
            custom_parameters: 自定义参数字典
        """
        # 验证和规范化输入
        self.adjust_valence = max(-1.0, min(1.0, adjust_valence))
        self.adjust_arousal = max(-1.0, min(1.0, adjust_arousal))
        self.reduce_leaps = reduce_leaps
        self.increase_stepwise = increase_stepwise
        self.motif_variation = max(0.0, min(1.0, motif_variation))
        self.tempo_adjustment = max(-1.0, min(1.0, tempo_adjustment))
        self.dynamics_range = max(0.0, min(1.0, dynamics_range))
        self.harmonic_complexity = max(0.0, min(1.0, harmonic_complexity))
        self.custom_parameters = custom_parameters or {}
        
        self.timestamp = time.time()
    
    @classmethod
    def from_user_input(cls, user_input: Dict[str, Any]) -> 'UserFeedback':
        """从用户输入字典创建反馈对象，带验证"""
        # 默认值字典
        defaults = {
            "adjust_valence": 0.0,
            "adjust_arousal": 0.0,
            "reduce_leaps": False,
            "increase_stepwise": False,
            "motif_variation": 0.0,
            "tempo_adjustment": 0.0,
            "dynamics_range": 0.5,
            "harmonic_complexity": 0.5
        }
        
        # 合并并验证输入
        validated_input = {}
        for key, default in defaults.items():
            if key in user_input:
                # 基本类型验证
                expected_type = type(default)
                if not isinstance(user_input[key], expected_type):
                    logger.warning(f"参数 {key} 类型错误，期望 {expected_type.__name__}，使用默认值 {default}")
                    validated_input[key] = default
                else:
                    validated_input[key] = user_input[key]
            else:
                validated_input[key] = default
        
        # 收集自定义参数
        custom_params = {k: v for k, v in user_input.items() if k not in defaults}
        validated_input["custom_parameters"] = custom_params
        
        return cls(**validated_input)
    
    def apply_to_emotion(self, emotion: PlutchikEmotion) -> PlutchikEmotion:
        """将用户反馈应用于情感状态"""
        valence, arousal = emotion.to_valence_arousal()
        
        # 调整 valence 和 arousal
        new_valence = DEFAULT_RANGE.validate(valence + self.adjust_valence)
        new_arousal = DEFAULT_RANGE.validate(arousal + self.adjust_arousal)
        
        # 通过逆映射调整 Plutchik 情感值
        return PlutchikEmotion.from_valence_arousal(new_valence, new_arousal, EmotionSource.USER_INPUT)


class EmotionManager:
    """情感管理器 - 中央系统，处理情感获取、存储和转换"""
    
    def __init__(self):
        self._emotion_cache = {}
        self._emotion_history = []
        self._emotion_providers = {}
        self._emotion_listeners = []
        self.current_emotion = PlutchikEmotion()  # 默认空情感
        
    def register_provider(self, name: str, provider: EmotionProvider) -> None:
        """注册情感提供者"""
        self._emotion_providers[name] = provider
        logger.info(f"注册情感提供者: {name}")
        
    def add_listener(self, callback: Callable[[PlutchikEmotion, EmotionChangeReason], None]) -> None:
        """添加情感变化监听器"""
        self._emotion_listeners.append(callback)
    
    def get_emotion(self, context: Any = None, provider_name: str = None) -> PlutchikEmotion:
        """获取情感状态"""
        # 如果指定了提供者，使用该提供者
        if provider_name and provider_name in self._emotion_providers:
            emotion = self._emotion_providers[provider_name].get_emotion(context)
            self._update_emotion(emotion, EmotionChangeReason.CONTEXT_SHIFT)
            return emotion
            
        # 使用map_context作为参数
        if context:
            # 检查缓存
            cache_key = str(hash(str(context)))
            if cache_key in self._emotion_cache:
                return self._emotion_cache[cache_key]
            
            # 从合适的提供者获取情感
            for provider in self._emotion_providers.values():
                try:
                    emotion = provider.get_emotion(context)
                    # 缓存结果
                    self._emotion_cache[cache_key] = emotion
                    self._update_emotion(emotion, EmotionChangeReason.CONTEXT_SHIFT)
                    return emotion
                except Exception as e:
                    logger.error(f"情感提供者错误: {e}")
                    continue
        
        # 返回当前情感状态
        return self.current_emotion
    
    def apply_user_feedback(self, feedback: UserFeedback) -> PlutchikEmotion:
        """应用用户反馈，调整当前情感状态"""
        new_emotion = feedback.apply_to_emotion(self.current_emotion)
        self._update_emotion(new_emotion, EmotionChangeReason.USER_ADJUSTMENT)
        return new_emotion
    
    def _update_emotion(self, emotion: PlutchikEmotion, reason: EmotionChangeReason) -> None:
        """更新当前情感状态并通知监听器"""
        old_emotion = self.current_emotion
        self.current_emotion = emotion
        
        # 保存到历史
        self._emotion_history.append((emotion, reason, time.time()))
        
        # 设置上下文变量
        current_emotion_ctx.set(emotion)
        
        # 通知监听器
        for listener in self._emotion_listeners:
            try:
                listener(emotion, reason)
            except Exception as e:
                logger.error(f"情感监听器错误: {e}")
    
    def save_emotion_state(self, filepath: str) -> None:
        """将当前情感状态保存到文件"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_emotion.to_dict(), f, indent=2)
            logger.info(f"情感状态已保存至 {filepath}")
        except Exception as e:
            logger.error(f"保存情感状态失败: {e}")
    
    def load_emotion_state(self, filepath: str) -> Optional[PlutchikEmotion]:
        """从文件加载情感状态"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            emotion = PlutchikEmotion.from_dict(data)
            self._update_emotion(emotion, EmotionChangeReason.INITIAL_SET)
            logger.info(f"从 {filepath} 加载情感状态")
            return emotion
        except Exception as e:
            logger.error(f"加载情感状态失败: {e}")
            return None


# 情感管理器单例
_emotion_manager = EmotionManager()

def get_emotion_manager() -> EmotionManager:
    """获取情感管理器单例"""
    return _emotion_manager

#####################################
# 获取情感信息
#####################################
def get_emotion_from_map(map_context: Any = None) -> PlutchikEmotion:
    """
    从映射上下文获取情感状态
    
    Args:
        map_context: 包含情感映射信息的上下文数据
        
    Returns:
        基于上下文的情感状态
    """
    # 首先尝试从管理器获取情感
    emotion_manager = get_emotion_manager()
    cached_emotion = emotion_manager.get_emotion(context=map_context)
    
    if cached_emotion is not None and isinstance(cached_emotion, PlutchikEmotion):
        logger.debug(f"从缓存获取情感: {cached_emotion.dominant_emotion}")
        return cached_emotion
        
    # 如果没有从管理器获取到，使用默认策略
    try:
        # 尝试从map_context中提取情感信息
        if isinstance(map_context, dict) and 'emotion_values' in map_context:
            values = map_context['emotion_values']
            return PlutchikEmotion(
                joy=values.get('joy', 0.5),
                trust=values.get('trust', 0.5),
                fear=values.get('fear', 0.2),
                surprise=values.get('surprise', 0.3),
                sadness=values.get('sadness', 0.1),
                disgust=values.get('disgust', 0.2),
                anger=values.get('anger', 0.1),
                anticipation=values.get('anticipation', 0.4),
                source=EmotionSource.PRESET
            )
    except Exception as e:
        logger.warning(f"提取情感数据失败: {e}，使用默认值")
    
    # 默认情感值
    default_emotion = PlutchikEmotion(
        joy=0.6,
        trust=0.5,
        fear=0.2,
        surprise=0.3,
        sadness=0.1,
        disgust=0.2,
        anger=0.1,
        anticipation=0.4,
        source=EmotionSource.PRESET
    )
    
    # 更新情感管理器
    emotion_manager._update_emotion(default_emotion, EmotionChangeReason.INITIAL_SET)
    
    return default_emotion


@lru_cache(maxsize=128)
def simulate_user_dialog(user_id: str = "default", session_id: str = None) -> Dict[str, Any]:
    """
    模拟或获取用户对音乐情感的反馈
    
    Args:
        user_id: 用户标识符，用于个性化
        session_id: 会话标识符，用于状态追踪
        
    Returns:
        包含用户反馈参数的字典
    """
    logger.debug(f"获取用户反馈: user_id={user_id}, session_id={session_id}")
    
    # 检查是否有持久化的用户偏好
    preferences_path = Path(f"./user_data/{user_id}/preferences.json")
    
    if preferences_path.exists():
        try:
            with open(preferences_path, 'r') as f:
                saved_prefs = json.load(f)
                logger.info(f"加载用户存储的偏好: {user_id}")
                return saved_prefs
        except Exception as e:
            logger.error(f"加载用户偏好失败: {e}")
    
    # 基于用户ID和会话ID生成一致性随机值
    if user_id and session_id:
        import hashlib
        seed = int(hashlib.md5(f"{user_id}:{session_id}".encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        # 根据用户ID生成轻微变化的偏好
        adjust_valence = np.random.uniform(-0.2, 0.2)
        adjust_arousal = np.random.uniform(-0.2, 0.2)
        reduce_leaps = np.random.random() > 0.5
        increase_stepwise = np.random.random() > 0.6
        motif_variation = np.random.uniform(0.0, 0.3)
        
        return {
            "adjust_valence": round(adjust_valence, 2),
            "adjust_arousal": round(adjust_arousal, 2),
            "reduce_leaps": reduce_leaps,
            "increase_stepwise": increase_stepwise,
            "motif_variation": round(motif_variation, 2),
            "tempo_adjustment": round(np.random.uniform(-0.3, 0.3), 2),
            "dynamics_range": round(np.random.uniform(0.3, 0.7), 2),
            "harmonic_complexity": round(np.random.uniform(0.3, 0.7), 2)
        }
    
    # 默认反馈值 - 兼容原代码
    return {
        "adjust_valence": 0.0,
        "adjust_arousal": 0.0,
        "reduce_leaps": True,
        "increase_stepwise": True,
        "motif_variation": 0.1,
        "tempo_adjustment": 0.0,
        "dynamics_range": 0.5,
        "harmonic_complexity": 0.5
    }