#####################################
# 和弦进行字典
#####################################
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from enum import Enum, auto
from dataclasses import dataclass, field
import re
import random
from functools import lru_cache
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

#####################################
# 核心音乐理论结构
#####################################

class NoteName(Enum):
    """音名枚举，支持所有基本音名及其变化形式"""
    C = 0
    C_SHARP = 1
    D_FLAT = 1
    D = 2
    D_SHARP = 3
    E_FLAT = 3
    E = 4
    F = 5
    F_SHARP = 6
    G_FLAT = 6
    G = 7
    G_SHARP = 8
    A_FLAT = 8
    A = 9
    A_SHARP = 10
    B_FLAT = 10
    B = 11
    
    @classmethod
    def from_string(cls, name: str) -> 'NoteName':
        """从字符串解析音名"""
        name = name.upper().replace('♯', '_SHARP').replace('#', '_SHARP')
        name = name.replace('♭', '_FLAT').replace('b', '_FLAT')
        try:
            return cls[name]
        except KeyError:
            # 尝试分离音名和变音记号
            if len(name) > 1 and name[1] in ('_', '#', 'b', '♯', '♭'):
                base = name[0]
                mod = '_SHARP' if name[1] in ('#', '♯') else '_FLAT'
                try:
                    return cls[f"{base}{mod}"]
                except KeyError:
                    raise ValueError(f"无法识别的音名: {name}")
            raise ValueError(f"无法识别的音名: {name}")
    
    def to_string(self, notation: str = 'standard') -> str:
        """将音名转换为字符串表示"""
        name = self.name.replace('_SHARP', '♯').replace('_FLAT', '♭')
        if notation == 'english':
            return name
        elif notation == 'standard':
            return name.replace('♯', '#').replace('♭', 'b')
        elif notation == 'german':
            mapping = {'B': 'H', 'B♭': 'B'}
            for orig, germ in mapping.items():
                if name == orig:
                    return germ
            return name
        return name
        
    def transpose(self, semitones: int) -> 'NoteName':
        """移调指定的半音数"""
        new_value = (self.value + semitones) % 12
        for name in NoteName:
            if name.value == new_value:
                return name
        return NoteName(new_value)  # 默认返回

class ChordQuality(Enum):
    """和弦质量/类型枚举"""
    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"
    MAJOR_7 = "maj7"
    DOMINANT_7 = "7"
    MINOR_7 = "min7"
    HALF_DIMINISHED = "m7b5"
    DIMINISHED_7 = "dim7"
    AUGMENTED_7 = "aug7"
    MAJOR_9 = "maj9"
    DOMINANT_9 = "9"
    MINOR_9 = "min9"
    ADD_9 = "add9"
    SIXTH = "6"
    MINOR_6 = "min6"
    # 更多和弦类型...
    
    @classmethod
    def from_symbol(cls, symbol: str) -> 'ChordQuality':
        """从和弦符号解析和弦类型"""
        # 映射常见的和弦符号到和弦类型
        mapping = {
            "": cls.MAJOR,       # 默认为大三和弦
            "M": cls.MAJOR,
            "maj": cls.MAJOR,
            "m": cls.MINOR,
            "min": cls.MINOR,
            "-": cls.MINOR,
            "dim": cls.DIMINISHED,
            "°": cls.DIMINISHED,
            "+": cls.AUGMENTED,
            "aug": cls.AUGMENTED,
            "sus2": cls.SUSPENDED_2,
            "sus4": cls.SUSPENDED_4,
            "sus": cls.SUSPENDED_4,
            "maj7": cls.MAJOR_7,
            "M7": cls.MAJOR_7,
            "7": cls.DOMINANT_7,
            "m7": cls.MINOR_7,
            "min7": cls.MINOR_7,
            "-7": cls.MINOR_7,
            "ø": cls.HALF_DIMINISHED,
            "m7b5": cls.HALF_DIMINISHED,
            "°7": cls.DIMINISHED_7,
            "dim7": cls.DIMINISHED_7,
            "+7": cls.AUGMENTED_7,
            "aug7": cls.AUGMENTED_7,
            "maj9": cls.MAJOR_9,
            "M9": cls.MAJOR_9,
            "9": cls.DOMINANT_9,
            "m9": cls.MINOR_9,
            "min9": cls.MINOR_9,
            "add9": cls.ADD_9,
            "6": cls.SIXTH,
            "m6": cls.MINOR_6,
            "min6": cls.MINOR_6
        }
        
        for key, value in mapping.items():
            if symbol.lower() == key.lower():
                return value
        
        raise ValueError(f"未知的和弦类型符号: {symbol}")
    
    def get_intervals(self) -> List[int]:
        """获取此和弦类型的音程结构（半音数）"""
        # 定义各种和弦类型的音程结构
        intervals = {
            ChordQuality.MAJOR: [0, 4, 7],           # 大三和弦: 根音、大三度、纯五度
            ChordQuality.MINOR: [0, 3, 7],           # 小三和弦: 根音、小三度、纯五度
            ChordQuality.DIMINISHED: [0, 3, 6],      # 减三和弦: 根音、小三度、减五度
            ChordQuality.AUGMENTED: [0, 4, 8],       # 增三和弦: 根音、大三度、增五度
            ChordQuality.SUSPENDED_2: [0, 2, 7],     # sus2和弦: 根音、大二度、纯五度
            ChordQuality.SUSPENDED_4: [0, 5, 7],     # sus4和弦: 根音、纯四度、纯五度
            ChordQuality.MAJOR_7: [0, 4, 7, 11],     # 大七和弦: 根音、大三度、纯五度、大七度
            ChordQuality.DOMINANT_7: [0, 4, 7, 10],  # 属七和弦: 根音、大三度、纯五度、小七度
            ChordQuality.MINOR_7: [0, 3, 7, 10],     # 小七和弦: 根音、小三度、纯五度、小七度
            ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],  # 半减七和弦: 根音、小三度、减五度、小七度
            ChordQuality.DIMINISHED_7: [0, 3, 6, 9], # 减七和弦: 根音、小三度、减五度、减七度
            ChordQuality.AUGMENTED_7: [0, 4, 8, 10], # 增七和弦: 根音、大三度、增五度、小七度
            ChordQuality.MAJOR_9: [0, 4, 7, 11, 14], # 大九和弦: 大七和弦加大九度
            ChordQuality.DOMINANT_9: [0, 4, 7, 10, 14], # 属九和弦: 属七和弦加大九度
            ChordQuality.MINOR_9: [0, 3, 7, 10, 14], # 小九和弦: 小七和弦加大九度
            ChordQuality.ADD_9: [0, 4, 7, 14],       # add9和弦: 大三和弦加大九度
            ChordQuality.SIXTH: [0, 4, 7, 9],        # 大六和弦: 大三和弦加大六度
            ChordQuality.MINOR_6: [0, 3, 7, 9],      # 小六和弦: 小三和弦加大六度
        }
        
        return intervals.get(self, [0, 4, 7])  # 默认为大三和弦
    
    def get_color_tension(self) -> Tuple[float, float]:
        """获取此和弦类型的色彩和张力特征向量"""
        # 定义各种和弦类型的主观色彩和张力值 (0-1范围)
        # 格式: (色彩值, 张力值) 其中:
        # 色彩: 0=暗, 0.5=中性, 1=亮
        # 张力: 0=松弛, 0.5=中等, 1=紧张
        color_tension = {
            ChordQuality.MAJOR: (0.8, 0.2),           # 明亮、稳定
            ChordQuality.MINOR: (0.3, 0.4),           # 暗淡、忧伤
            ChordQuality.DIMINISHED: (0.2, 0.8),      # 暗淡、不稳定
            ChordQuality.AUGMENTED: (0.6, 0.7),       # 闪亮、紧张
            ChordQuality.SUSPENDED_2: (0.6, 0.5),     # 明亮、悬而未决
            ChordQuality.SUSPENDED_4: (0.5, 0.6),     # 中性、悬而未决
            ChordQuality.MAJOR_7: (0.9, 0.3),         # 非常明亮、温暖
            ChordQuality.DOMINANT_7: (0.7, 0.6),      # 明亮、有紧张感
            ChordQuality.MINOR_7: (0.4, 0.4),         # 暗淡、微忧伤
            ChordQuality.HALF_DIMINISHED: (0.3, 0.7), # 暗淡、紧张
            ChordQuality.DIMINISHED_7: (0.2, 0.8),    # 很暗、非常紧张
            ChordQuality.AUGMENTED_7: (0.5, 0.9),     # 中性、极度紧张
            ChordQuality.MAJOR_9: (0.9, 0.4),         # 非常明亮、温暖但复杂
            ChordQuality.DOMINANT_9: (0.7, 0.7),      # 明亮、复杂紧张
            ChordQuality.MINOR_9: (0.5, 0.5),         # 中性、复杂忧伤
            ChordQuality.ADD_9: (0.8, 0.4),           # 明亮、轻度复杂
            ChordQuality.SIXTH: (0.8, 0.3),           # 明亮、温暖
            ChordQuality.MINOR_6: (0.5, 0.4),         # 中性、温暖忧伤
        }
        
        return color_tension.get(self, (0.5, 0.5))  # 默认为中性、中等张力

class HarmonicFunction(Enum):
    """和声功能枚举，表示和弦在调性中的角色"""
    TONIC = 1              # 主和弦 (I级)
    SUPERTONIC = 2         # 上主和弦 (II级)
    MEDIANT = 3            # 中音和弦 (III级)
    SUBDOMINANT = 4        # 下属和弦 (IV级) 
    DOMINANT = 5           # 属和弦 (V级)
    SUBMEDIANT = 6         # 下中音和弦 (VI级)
    LEADING_TONE = 7       # 导音和弦 (VII级)
    SECONDARY_DOMINANT = 8 # 副属和弦
    BORROWED = 9           # 借用和弦 
    PASSING = 10           # 经过和弦
    CADENTIAL = 11         # 终止和弦
    PEDAL = 12             # 持续音和弦

class ScaleType(Enum):
    """音阶类型枚举，定义常见和特殊的音阶"""
    # 西方音阶
    MAJOR = "major"                 # 自然大调
    NATURAL_MINOR = "natural_minor" # 自然小调
    HARMONIC_MINOR = "harmonic_minor" # 和声小调
    MELODIC_MINOR = "melodic_minor" # 旋律小调
    DORIAN = "dorian"               # 多利亚调式
    PHRYGIAN = "phrygian"           # 弗里几亚调式
    LYDIAN = "lydian"               # 利底亚调式
    MIXOLYDIAN = "mixolydian"       # 混合利底亚调式
    AEOLIAN = "aeolian"             # 爱奥利亚调式(自然小调)
    LOCRIAN = "locrian"             # 洛克里亚调式
    
    # 中国五声音阶
    CHINESE_PENTATONIC = "chinese_pentatonic" # 中国五声音阶(宫商角徵羽)
    GONG = "gong"                   # 宫调式
    SHANG = "shang"                 # 商调式
    JUE = "jue"                     # 角调式
    ZHI = "zhi"                     # 徵调式
    YU = "yu"                       # 羽调式
    
    # 印度音阶
    INDIAN_RAGA_BHAIRAV = "raga_bhairav" # 拜拉夫拉格
    INDIAN_RAGA_KALYAN = "raga_kalyan"   # 卡尔扬拉格
    
    # 阿拉伯音阶
    ARABIC_MAQAM_RAST = "maqam_rast"  # 拉斯特调式
    ARABIC_MAQAM_BAYATI = "maqam_bayati" # 巴亚提调式
    
    # 非洲音阶
    AFRICAN_KUMOI = "african_kumoi"  # 库莫伊音阶
    
    # 日本音阶
    JAPANESE_IN = "japanese_in"    # 日本音阶
    
    # 爵士和现代音阶
    BLUES = "blues"               # 布鲁斯音阶
    BEBOP = "bebop"               # 比博普音阶
    DIMINISHED = "diminished"     # 减音阶
    WHOLE_TONE = "whole_tone"     # 全音音阶
    CHROMATIC = "chromatic"       # 半音音阶
    
    def get_intervals(self) -> List[int]:
        """获取此音阶类型的音程结构（半音数）"""
        intervals = {
            # 西方调式音阶
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.NATURAL_MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
            ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
            ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
            ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
            ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            ScaleType.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
            
            # 中国五声音阶
            ScaleType.CHINESE_PENTATONIC: [0, 2, 4, 7, 9],
            ScaleType.GONG: [0, 2, 4, 7, 9],
            ScaleType.SHANG: [0, 2, 5, 7, 10],
            ScaleType.JUE: [0, 3, 5, 8, 10],
            ScaleType.ZHI: [0, 2, 5, 7, 10],
            ScaleType.YU: [0, 3, 5, 8, 10],
            
            # 印度音阶
            ScaleType.INDIAN_RAGA_BHAIRAV: [0, 1, 4, 5, 7, 8, 11],
            ScaleType.INDIAN_RAGA_KALYAN: [0, 2, 4, 6, 7, 9, 11],
            
            # 阿拉伯音阶
            ScaleType.ARABIC_MAQAM_RAST: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.ARABIC_MAQAM_BAYATI: [0, 1, 3, 5, 7, 8, 10],
            
            # 非洲音阶
            ScaleType.AFRICAN_KUMOI: [0, 2, 3, 7, 9],
            
            # 日本音阶
            ScaleType.JAPANESE_IN: [0, 1, 5, 7, 8],
            
            # 爵士和现代音阶
            ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
            ScaleType.BEBOP: [0, 2, 4, 5, 7, 9, 10, 11],
            ScaleType.DIMINISHED: [0, 2, 3, 5, 6, 8, 9, 11],
            ScaleType.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
            ScaleType.CHROMATIC: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
        
        return intervals.get(self, [0, 2, 4, 5, 7, 9, 11])  # 默认为大调音阶

class CulturalSystem(Enum):
    """定义全球音乐文化系统"""
    WESTERN = "western"     # 西方古典和现代音乐
    CHINESE = "chinese"     # 中国传统音乐 
    INDIAN = "indian"       # 印度音乐
    ARABIC = "arabic"       # 阿拉伯音乐
    AFRICAN = "african"     # 非洲音乐
    JAPANESE = "japanese"   # 日本音乐
    LATIN = "latin"         # 拉丁美洲音乐
    CELTIC = "celtic"       # 凯尔特音乐
    BALKAN = "balkan"       # 巴尔干音乐
    MICROTONAL = "microtonal" # 微分音系统
    
    @classmethod
    def get_native_scales(cls, culture: 'CulturalSystem') -> List[ScaleType]:
        """获取特定文化常用的音阶类型"""
        scales = {
            CulturalSystem.WESTERN: [
                ScaleType.MAJOR, ScaleType.NATURAL_MINOR, ScaleType.HARMONIC_MINOR, 
                ScaleType.MELODIC_MINOR, ScaleType.DORIAN, ScaleType.PHRYGIAN,
                ScaleType.LYDIAN, ScaleType.MIXOLYDIAN, ScaleType.AEOLIAN, ScaleType.LOCRIAN
            ],
            CulturalSystem.CHINESE: [
                ScaleType.CHINESE_PENTATONIC, ScaleType.GONG, ScaleType.SHANG, 
                ScaleType.JUE, ScaleType.ZHI, ScaleType.YU
            ],
            CulturalSystem.INDIAN: [
                ScaleType.INDIAN_RAGA_BHAIRAV, ScaleType.INDIAN_RAGA_KALYAN
            ],
            CulturalSystem.ARABIC: [
                ScaleType.ARABIC_MAQAM_RAST, ScaleType.ARABIC_MAQAM_BAYATI
            ],
            CulturalSystem.AFRICAN: [
                ScaleType.AFRICAN_KUMOI, ScaleType.BLUES
            ],
            CulturalSystem.JAPANESE: [
                ScaleType.JAPANESE_IN
            ],
        }
        
        return scales.get(culture, [ScaleType.MAJOR, ScaleType.NATURAL_MINOR])
    
    @classmethod
    def get_characteristics(cls, culture: 'CulturalSystem') -> Dict[str, Any]:
        """获取特定文化的音乐特征参数"""
        characteristics = {
            CulturalSystem.WESTERN: {
                "voice_leading_emphasis": 0.8,  # 声部进行重要性
                "harmonic_complexity": 0.7,     # 和声复杂度
                "typical_rhythms": ["4/4", "3/4", "6/8"],
                "ornament_density": 0.3,        # 装饰音密度
                "microtonal": False,            # 是否使用微分音
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.MINOR, 
                                             ChordQuality.DOMINANT_7]
            },
            CulturalSystem.CHINESE: {
                "voice_leading_emphasis": 0.4,
                "harmonic_complexity": 0.4,
                "typical_rhythms": ["4/4", "2/4"],
                "ornament_density": 0.7,
                "microtonal": False,
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.SUSPENDED_2, 
                                             ChordQuality.SUSPENDED_4]
            },
            CulturalSystem.INDIAN: {
                "voice_leading_emphasis": 0.3,
                "harmonic_complexity": 0.5,
                "typical_rhythms": ["7/8", "10/8", "16/8"],
                "ornament_density": 0.9,
                "microtonal": True,
                "preferred_chord_qualities": [ChordQuality.SUSPENDED_4, ChordQuality.ADD_9]
            },
            CulturalSystem.ARABIC: {
                "voice_leading_emphasis": 0.2,
                "harmonic_complexity": 0.6,
                "typical_rhythms": ["10/8", "7/8"],
                "ornament_density": 0.8,
                "microtonal": True,
                "preferred_chord_qualities": [ChordQuality.SUSPENDED_2, ChordQuality.AUGMENTED]
            },
            CulturalSystem.AFRICAN: {
                "voice_leading_emphasis": 0.5,
                "harmonic_complexity": 0.4,
                "typical_rhythms": ["12/8", "6/8", "9/8"],
                "ornament_density": 0.6,
                "microtonal": False,
                "polyrhythmic": True,
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.SUSPENDED_4]
            },
        }
        
        return characteristics.get(culture, {"voice_leading_emphasis": 0.5, "harmonic_complexity": 0.5})

@dataclass
class Note:
    """表示单个音符，包含音名、八度和精确频率信息"""
    name: NoteName
    octave: int
    
    @property
    def midi_number(self) -> int:
        """转换为MIDI音符编号"""
        return self.octave * 12 + self.name.value + 12
    
    @classmethod
    def from_string(cls, note_str: str) -> 'Note':
        """从字符串解析音符"""
        # 匹配如 "C4", "F#5", "Gb3" 格式的音符
        match = re.match(r'([A-Ga-g][#b♯♭]?)(\d+)', note_str)
        if match:
            name_str, octave_str = match.groups()
            try:
                note_name = NoteName.from_string(name_str)
                octave = int(octave_str)
                return cls(note_name, octave)
            except (ValueError, KeyError) as e:
                raise ValueError(f"无效的音符格式 '{note_str}': {e}")
        else:
            # 处理和弦格式如 "Cm3", "F#maj7"
            chord_match = re.match(r'([A-Ga-g][#b♯import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from enum import Enum, auto
from dataclasses import dataclass, field
import re
import random
from functools import lru_cache
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

#####################################
# 核心音乐理论结构
#####################################

class NoteName(Enum):
    """音名枚举，支持所有基本音名及其变化形式"""
    C = 0
    C_SHARP = 1
    D_FLAT = 1
    D = 2
    D_SHARP = 3
    E_FLAT = 3
    E = 4
    F = 5
    F_SHARP = 6
    G_FLAT = 6
    G = 7
    G_SHARP = 8
    A_FLAT = 8
    A = 9
    A_SHARP = 10
    B_FLAT = 10
    B = 11
    
    @classmethod
    def from_string(cls, name: str) -> 'NoteName':
        """从字符串解析音名"""
        name = name.upper().replace('♯', '_SHARP').replace('#', '_SHARP')
        name = name.replace('♭', '_FLAT').replace('b', '_FLAT')
        try:
            return cls[name]
        except KeyError:
            # 尝试分离音名和变音记号
            if len(name) > 1 and name[1] in ('_', '#', 'b', '♯', '♭'):
                base = name[0]
                mod = '_SHARP' if name[1] in ('#', '♯') else '_FLAT'
                try:
                    return cls[f"{base}{mod}"]
                except KeyError:
                    raise ValueError(f"无法识别的音名: {name}")
            raise ValueError(f"无法识别的音名: {name}")
    
    def to_string(self, notation: str = 'standard') -> str:
        """将音名转换为字符串表示"""
        name = self.name.replace('_SHARP', '♯').replace('_FLAT', '♭')
        if notation == 'english':
            return name
        elif notation == 'standard':
            return name.replace('♯', '#').replace('♭', 'b')
        elif notation == 'german':
            mapping = {'B': 'H', 'B♭': 'B'}
            for orig, germ in mapping.items():
                if name == orig:
                    return germ
            return name
        return name
        
    def transpose(self, semitones: int) -> 'NoteName':
        """移调指定的半音数"""
        new_value = (self.value + semitones) % 12
        for name in NoteName:
            if name.value == new_value:
                return name
        return NoteName(new_value)  # 默认返回

class ChordQuality(Enum):
    """和弦质量/类型枚举"""
    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"
    MAJOR_7 = "maj7"
    DOMINANT_7 = "7"
    MINOR_7 = "min7"
    HALF_DIMINISHED = "m7b5"
    DIMINISHED_7 = "dim7"
    AUGMENTED_7 = "aug7"
    MAJOR_9 = "maj9"
    DOMINANT_9 = "9"
    MINOR_9 = "min9"
    ADD_9 = "add9"
    SIXTH = "6"
    MINOR_6 = "min6"
    # 更多和弦类型...
    
    @classmethod
    def from_symbol(cls, symbol: str) -> 'ChordQuality':
        """从和弦符号解析和弦类型"""
        # 映射常见的和弦符号到和弦类型
        mapping = {
            "": cls.MAJOR,       # 默认为大三和弦
            "M": cls.MAJOR,
            "maj": cls.MAJOR,
            "m": cls.MINOR,
            "min": cls.MINOR,
            "-": cls.MINOR,
            "dim": cls.DIMINISHED,
            "°": cls.DIMINISHED,
            "+": cls.AUGMENTED,
            "aug": cls.AUGMENTED,
            "sus2": cls.SUSPENDED_2,
            "sus4": cls.SUSPENDED_4,
            "sus": cls.SUSPENDED_4,
            "maj7": cls.MAJOR_7,
            "M7": cls.MAJOR_7,
            "7": cls.DOMINANT_7,
            "m7": cls.MINOR_7,
            "min7": cls.MINOR_7,
            "-7": cls.MINOR_7,
            "ø": cls.HALF_DIMINISHED,
            "m7b5": cls.HALF_DIMINISHED,
            "°7": cls.DIMINISHED_7,
            "dim7": cls.DIMINISHED_7,
            "+7": cls.AUGMENTED_7,
            "aug7": cls.AUGMENTED_7,
            "maj9": cls.MAJOR_9,
            "M9": cls.MAJOR_9,
            "9": cls.DOMINANT_9,
            "m9": cls.MINOR_9,
            "min9": cls.MINOR_9,
            "add9": cls.ADD_9,
            "6": cls.SIXTH,
            "m6": cls.MINOR_6,
            "min6": cls.MINOR_6
        }
        
        for key, value in mapping.items():
            if symbol.lower() == key.lower():
                return value
        
        raise ValueError(f"未知的和弦类型符号: {symbol}")
    
    def get_intervals(self) -> List[int]:
        """获取此和弦类型的音程结构（半音数）"""
        # 定义各种和弦类型的音程结构
        intervals = {
            ChordQuality.MAJOR: [0, 4, 7],           # 大三和弦: 根音、大三度、纯五度
            ChordQuality.MINOR: [0, 3, 7],           # 小三和弦: 根音、小三度、纯五度
            ChordQuality.DIMINISHED: [0, 3, 6],      # 减三和弦: 根音、小三度、减五度
            ChordQuality.AUGMENTED: [0, 4, 8],       # 增三和弦: 根音、大三度、增五度
            ChordQuality.SUSPENDED_2: [0, 2, 7],     # sus2和弦: 根音、大二度、纯五度
            ChordQuality.SUSPENDED_4: [0, 5, 7],     # sus4和弦: 根音、纯四度、纯五度
            ChordQuality.MAJOR_7: [0, 4, 7, 11],     # 大七和弦: 根音、大三度、纯五度、大七度
            ChordQuality.DOMINANT_7: [0, 4, 7, 10],  # 属七和弦: 根音、大三度、纯五度、小七度
            ChordQuality.MINOR_7: [0, 3, 7, 10],     # 小七和弦: 根音、小三度、纯五度、小七度
            ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],  # 半减七和弦: 根音、小三度、减五度、小七度
            ChordQuality.DIMINISHED_7: [0, 3, 6, 9], # 减七和弦: 根音、小三度、减五度、减七度
            ChordQuality.AUGMENTED_7: [0, 4, 8, 10], # 增七和弦: 根音、大三度、增五度、小七度
            ChordQuality.MAJOR_9: [0, 4, 7, 11, 14], # 大九和弦: 大七和弦加大九度
            ChordQuality.DOMINANT_9: [0, 4, 7, 10, 14], # 属九和弦: 属七和弦加大九度
            ChordQuality.MINOR_9: [0, 3, 7, 10, 14], # 小九和弦: 小七和弦加大九度
            ChordQuality.ADD_9: [0, 4, 7, 14],       # add9和弦: 大三和弦加大九度
            ChordQuality.SIXTH: [0, 4, 7, 9],        # 大六和弦: 大三和弦加大六度
            ChordQuality.MINOR_6: [0, 3, 7, 9],      # 小六和弦: 小三和弦加大六度
        }
        
        return intervals.get(self, [0, 4, 7])  # 默认为大三和弦
    
    def get_color_tension(self) -> Tuple[float, float]:
        """获取此和弦类型的色彩和张力特征向量"""
        # 定义各种和弦类型的主观色彩和张力值 (0-1范围)
        # 格式: (色彩值, 张力值) 其中:
        # 色彩: 0=暗, 0.5=中性, 1=亮
        # 张力: 0=松弛, 0.5=中等, 1=紧张
        color_tension = {
            ChordQuality.MAJOR: (0.8, 0.2),           # 明亮、稳定
            ChordQuality.MINOR: (0.3, 0.4),           # 暗淡、忧伤
            ChordQuality.DIMINISHED: (0.2, 0.8),      # 暗淡、不稳定
            ChordQuality.AUGMENTED: (0.6, 0.7),       # 闪亮、紧张
            ChordQuality.SUSPENDED_2: (0.6, 0.5),     # 明亮、悬而未决
            ChordQuality.SUSPENDED_4: (0.5, 0.6),     # 中性、悬而未决
            ChordQuality.MAJOR_7: (0.9, 0.3),         # 非常明亮、温暖
            ChordQuality.DOMINANT_7: (0.7, 0.6),      # 明亮、有紧张感
            ChordQuality.MINOR_7: (0.4, 0.4),         # 暗淡、微忧伤
            ChordQuality.HALF_DIMINISHED: (0.3, 0.7), # 暗淡、紧张
            ChordQuality.DIMINISHED_7: (0.2, 0.8),    # 很暗、非常紧张
            ChordQuality.AUGMENTED_7: (0.5, 0.9),     # 中性、极度紧张
            ChordQuality.MAJOR_9: (0.9, 0.4),         # 非常明亮、温暖但复杂
            ChordQuality.DOMINANT_9: (0.7, 0.7),      # 明亮、复杂紧张
            ChordQuality.MINOR_9: (0.5, 0.5),         # 中性、复杂忧伤
            ChordQuality.ADD_9: (0.8, 0.4),           # 明亮、轻度复杂
            ChordQuality.SIXTH: (0.8, 0.3),           # 明亮、温暖
            ChordQuality.MINOR_6: (0.5, 0.4),         # 中性、温暖忧伤
        }
        
        return color_tension.get(self, (0.5, 0.5))  # 默认为中性、中等张力

class HarmonicFunction(Enum):
    """和声功能枚举，表示和弦在调性中的角色"""
    TONIC = 1              # 主和弦 (I级)
    SUPERTONIC = 2         # 上主和弦 (II级)
    MEDIANT = 3            # 中音和弦 (III级)
    SUBDOMINANT = 4        # 下属和弦 (IV级) 
    DOMINANT = 5           # 属和弦 (V级)
    SUBMEDIANT = 6         # 下中音和弦 (VI级)
    LEADING_TONE = 7       # 导音和弦 (VII级)
    SECONDARY_DOMINANT = 8 # 副属和弦
    BORROWED = 9           # 借用和弦 
    PASSING = 10           # 经过和弦
    CADENTIAL = 11         # 终止和弦
    PEDAL = 12             # 持续音和弦

class ScaleType(Enum):
    """音阶类型枚举，定义常见和特殊的音阶"""
    # 西方音阶
    MAJOR = "major"                 # 自然大调
    NATURAL_MINOR = "natural_minor" # 自然小调
    HARMONIC_MINOR = "harmonic_minor" # 和声小调
    MELODIC_MINOR = "melodic_minor" # 旋律小调
    DORIAN = "dorian"               # 多利亚调式
    PHRYGIAN = "phrygian"           # 弗里几亚调式
    LYDIAN = "lydian"               # 利底亚调式
    MIXOLYDIAN = "mixolydian"       # 混合利底亚调式
    AEOLIAN = "aeolian"             # 爱奥利亚调式(自然小调)
    LOCRIAN = "locrian"             # 洛克里亚调式
    
    # 中国五声音阶
    CHINESE_PENTATONIC = "chinese_pentatonic" # 中国五声音阶(宫商角徵羽)
    GONG = "gong"                   # 宫调式
    SHANG = "shang"                 # 商调式
    JUE = "jue"                     # 角调式
    ZHI = "zhi"                     # 徵调式
    YU = "yu"                       # 羽调式
    
    # 印度音阶
    INDIAN_RAGA_BHAIRAV = "raga_bhairav" # 拜拉夫拉格
    INDIAN_RAGA_KALYAN = "raga_kalyan"   # 卡尔扬拉格
    
    # 阿拉伯音阶
    ARABIC_MAQAM_RAST = "maqam_rast"  # 拉斯特调式
    ARABIC_MAQAM_BAYATI = "maqam_bayati" # 巴亚提调式
    
    # 非洲音阶
    AFRICAN_KUMOI = "african_kumoi"  # 库莫伊音阶
    
    # 日本音阶
    JAPANESE_IN = "japanese_in"    # 日本音阶
    
    # 爵士和现代音阶
    BLUES = "blues"               # 布鲁斯音阶
    BEBOP = "bebop"               # 比博普音阶
    DIMINISHED = "diminished"     # 减音阶
    WHOLE_TONE = "whole_tone"     # 全音音阶
    CHROMATIC = "chromatic"       # 半音音阶
    
    def get_intervals(self) -> List[int]:
        """获取此音阶类型的音程结构（半音数）"""
        intervals = {
            # 西方调式音阶
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.NATURAL_MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
            ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
            ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
            ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
            ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            ScaleType.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
            
            # 中国五声音阶
            ScaleType.CHINESE_PENTATONIC: [0, 2, 4, 7, 9],
            ScaleType.GONG: [0, 2, 4, 7, 9],
            ScaleType.SHANG: [0, 2, 5, 7, 10],
            ScaleType.JUE: [0, 3, 5, 8, 10],
            ScaleType.ZHI: [0, 2, 5, 7, 10],
            ScaleType.YU: [0, 3, 5, 8, 10],
            
            # 印度音阶
            ScaleType.INDIAN_RAGA_BHAIRAV: [0, 1, 4, 5, 7, 8, 11],
            ScaleType.INDIAN_RAGA_KALYAN: [0, 2, 4, 6, 7, 9, 11],
            
            # 阿拉伯音阶
            ScaleType.ARABIC_MAQAM_RAST: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.ARABIC_MAQAM_BAYATI: [0, 1, 3, 5, 7, 8, 10],
            
            # 非洲音阶
            ScaleType.AFRICAN_KUMOI: [0, 2, 3, 7, 9],
            
            # 日本音阶
            ScaleType.JAPANESE_IN: [0, 1, 5, 7, 8],
            
            # 爵士和现代音阶
            ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
            ScaleType.BEBOP: [0, 2, 4, 5, 7, 9, 10, 11],
            ScaleType.DIMINISHED: [0, 2, 3, 5, 6, 8, 9, 11],
            ScaleType.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
            ScaleType.CHROMATIC: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
        
        return intervals.get(self, [0, 2, 4, 5, 7, 9, 11])  # 默认为大调音阶

class CulturalSystem(Enum):
    """定义全球音乐文化系统"""
    WESTERN = "western"     # 西方古典和现代音乐
    CHINESE = "chinese"     # 中国传统音乐 
    INDIAN = "indian"       # 印度音乐
    ARABIC = "arabic"       # 阿拉伯音乐
    AFRICAN = "african"     # 非洲音乐
    JAPANESE = "japanese"   # 日本音乐
    LATIN = "latin"         # 拉丁美洲音乐
    CELTIC = "celtic"       # 凯尔特音乐
    BALKAN = "balkan"       # 巴尔干音乐
    MICROTONAL = "microtonal" # 微分音系统
    
    @classmethod
    def get_native_scales(cls, culture: 'CulturalSystem') -> List[ScaleType]:
        """获取特定文化常用的音阶类型"""
        scales = {
            CulturalSystem.WESTERN: [
                ScaleType.MAJOR, ScaleType.NATURAL_MINOR, ScaleType.HARMONIC_MINOR, 
                ScaleType.MELODIC_MINOR, ScaleType.DORIAN, ScaleType.PHRYGIAN,
                ScaleType.LYDIAN, ScaleType.MIXOLYDIAN, ScaleType.AEOLIAN, ScaleType.LOCRIAN
            ],
            CulturalSystem.CHINESE: [
                ScaleType.CHINESE_PENTATONIC, ScaleType.GONG, ScaleType.SHANG, 
                ScaleType.JUE, ScaleType.ZHI, ScaleType.YU
            ],
            CulturalSystem.INDIAN: [
                ScaleType.INDIAN_RAGA_BHAIRAV, ScaleType.INDIAN_RAGA_KALYAN
            ],
            CulturalSystem.ARABIC: [
                ScaleType.ARABIC_MAQAM_RAST, ScaleType.ARABIC_MAQAM_BAYATI
            ],
            CulturalSystem.AFRICAN: [
                ScaleType.AFRICAN_KUMOI, ScaleType.BLUES
            ],
            CulturalSystem.JAPANESE: [
                ScaleType.JAPANESE_IN
            ],
        }
        
        return scales.get(culture, [ScaleType.MAJOR, ScaleType.NATURAL_MINOR])
    
    @classmethod
    def get_characteristics(cls, culture: 'CulturalSystem') -> Dict[str, Any]:
        """获取特定文化的音乐特征参数"""
        characteristics = {
            CulturalSystem.WESTERN: {
                "voice_leading_emphasis": 0.8,  # 声部进行重要性
                "harmonic_complexity": 0.7,     # 和声复杂度
                "typical_rhythms": ["4/4", "3/4", "6/8"],
                "ornament_density": 0.3,        # 装饰音密度
                "microtonal": False,            # 是否使用微分音
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.MINOR, 
                                             ChordQuality.DOMINANT_7]
            },
            CulturalSystem.CHINESE: {
                "voice_leading_emphasis": 0.4,
                "harmonic_complexity": 0.4,
                "typical_rhythms": ["4/4", "2/4"],
                "ornament_density": 0.7,
                "microtonal": False,
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.SUSPENDED_2, 
                                             ChordQuality.SUSPENDED_4]
            },
            CulturalSystem.INDIAN: {
                "voice_leading_emphasis": 0.3,
                "harmonic_complexity": 0.5,
                "typical_rhythms": ["7/8", "10/8", "16/8"],
                "ornament_density": 0.9,
                "microtonal": True,
                "preferred_chord_qualities": [ChordQuality.SUSPENDED_4, ChordQuality.ADD_9]
            },
            CulturalSystem.ARABIC: {
                "voice_leading_emphasis": 0.2,
                "harmonic_complexity": 0.6,
                "typical_rhythms": ["10/8", "7/8"],
                "ornament_density": 0.8,
                "microtonal": True,
                "preferred_chord_qualities": [ChordQuality.SUSPENDED_2, ChordQuality.AUGMENTED]
            },
            CulturalSystem.AFRICAN: {
                "voice_leading_emphasis": 0.5,
                "harmonic_complexity": 0.4,
                "typical_rhythms": ["12/8", "6/8", "9/8"],
                "ornament_density": 0.6,
                "microtonal": False,
                "polyrhythmic": True,
                "preferred_chord_qualities": [ChordQuality.MAJOR, ChordQuality.SUSPENDED_4]
            },
        }
        
        return characteristics.get(culture, {"voice_leading_emphasis": 0.5, "harmonic_complexity": 0.5})

@dataclass
class Note:
    """表示单个音符，包含音名、八度和精确频率信息"""
    name: NoteName
    octave: int
    
    @property
    def midi_number(self) -> int:
        """转换为MIDI音符编号"""
        return self.octave * 12 + self.name.value + 12
    
    @classmethod
    def from_string(cls, note_str: str) -> 'Note':
        """从字符串解析音符"""
        # 匹配如 "C4", "F#5", "Gb3" 格式的音符
        match = re.match(r'([A-Ga-g][#b♯♭]?)(\d+)', note_str)
        if match:
            name_str, octave_str = match.groups()
            try:
                note_name = NoteName.from_string(name_str)
                octave = int(octave_str)
                return cls(note_name, octave)
            except (ValueError, KeyError) as e:
                raise ValueError(f"无效的音符格式 '{note_str}': {e}")
        else:
            # 尝试匹配和弦格式如 "Cm3", "F#maj7" 等，不在此处处理
            chord_match = re.match(r'([A-Ga-g][#b♯♭]?)(.*)', note_str)
            if chord_match:
                # 检测到和弦符号，直接提示不支持
                raise ValueError(f"'{note_str}' 为和弦形式，不能作为单个 Note 解析")
            
            raise ValueError(f"无法解析 '{note_str}'，未知的音符或和弦格式")