#####################################
# 定义文化设置
#####################################
import enum
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, Union, Any, TypeVar, Generic, Callable
import logging
from pathlib import Path
from functools import lru_cache
import numpy as np
import copy
from abc import ABC, abstractmethod
import re
from collections import defaultdict
import threading

#####################################
# 核心音乐理论模型
#####################################

class Interval(Enum):
    """精确的音程定义，支持微分音程"""
    PERFECT_UNISON = 0
    MINOR_SECOND = 1
    MAJOR_SECOND = 2
    MINOR_THIRD = 3
    MAJOR_THIRD = 4
    PERFECT_FOURTH = 5
    TRITONE = 6
    PERFECT_FIFTH = 7
    MINOR_SIXTH = 8
    MAJOR_SIXTH = 9
    MINOR_SEVENTH = 10
    MAJOR_SEVENTH = 11
    PERFECT_OCTAVE = 12
    
    # 微分音程
    QUARTER_TONE = 0.5
    THREE_QUARTER_TONE = 1.5
    FIVE_QUARTER_TONE = 2.5
    SEVEN_QUARTER_TONE = 3.5
    NEUTRAL_THIRD = 3.5  # 中性三度(介于小三和大三之间)
    NEUTRAL_SIXTH = 8.5  # 中性六度
    NEUTRAL_SEVENTH = 10.5  # 中性七度
    
    # 扩展音程(用于特定文化)
    SHRUTI_22_TONE = 0.55  # 印度22平均律系统音程
    PELOG_STEP = 2.4  # 印尼爪哇佩洛格(Pelog)音阶标准步进
    
    @classmethod
    def from_semitones(cls, semitones: float) -> 'Interval':
        """从半音数创建最接近的音程"""
        if isinstance(semitones, int):
            try:
                return cls(semitones)
            except ValueError:
                pass
        
        # 处理微分音程，寻找最接近值
        min_diff = float('inf')
        closest = cls.PERFECT_UNISON
        
        for interval in cls:
            diff = abs(interval.value - semitones)
            if diff < min_diff:
                min_diff = diff
                closest = interval
                
        return closest

class TuningSystem(Enum):
    """音律系统"""
    EQUAL_TEMPERAMENT = auto()       # 十二平均律
    PYTHAGOREAN = auto()             # 毕达哥拉斯律
    JUST_INTONATION = auto()         # 纯律
    MEANTONE = auto()                # 中庸律
    WELL_TEMPERED = auto()           # 良律
    INDIAN_22_SHRUTI = auto()        # 印度22平均分割系统
    ARABIC_24_TONE = auto()          # 阿拉伯24平均分割系统 
    INDONESIAN_SLENDRO = auto()      # 印尼五声音阶(斯伦德罗)
    INDONESIAN_PELOG = auto()        # 印尼七声音阶(佩洛格)
    CHINESE_PENTATONIC = auto()      # 中国五声律制
    AFRICAN_EQUIHEPTATONIC = auto()  # 非洲七平均律系统

class Mode(Enum):
    """调式类型"""
    # 西方传统
    IONIAN = auto()       # 伊奥尼亚调式(大调式)
    DORIAN = auto()       # 多利亚调式
    PHRYGIAN = auto()     # 弗里几亚调式
    LYDIAN = auto()       # 利底亚调式
    MIXOLYDIAN = auto()   # 混合利底亚调式
    AEOLIAN = auto()      # 爱奥尼亚调式(小调式)
    LOCRIAN = auto()      # 洛克里亚调式
    
    # 和声与旋律
    HARMONIC_MINOR = auto()  # 和声小调
    MELODIC_MINOR = auto()   # 旋律小调
    
    # 五声音阶变体
    PENTATONIC_MAJOR = auto()  # 大调式五声音阶
    PENTATONIC_MINOR = auto()  # 小调式五声音阶
    
    # 印度调式
    RAGA_BHAIRAV = auto()     # 拜拉夫拉伽
    RAGA_TODI = auto()        # 托迪拉伽
    RAGA_PURVI = auto()       # 普尔维拉伽
    
    # 阿拉伯马卡姆
    MAQAM_RAST = auto()       # 拉斯特调式
    MAQAM_BAYATI = auto()     # 巴亚提调式
    MAQAM_HIJAZ = auto()      # 希贾兹调式
    
    # 东亚调式
    CHINESE_GONG = auto()     # 宫调式
    CHINESE_SHANG = auto()    # 商调式
    CHINESE_JUE = auto()      # 角调式
    CHINESE_ZHI = auto()      # 徵调式
    CHINESE_YU = auto()       # 羽调式
    
    JAPANESE_IN = auto()      # 日本阴调式
    JAPANESE_YO = auto()      # 日本阳调式
    
    # 非洲调式
    AFRICAN_KUMOI = auto()    # 库莫伊调式
    AFRICAN_BEBOP = auto()    # 比博普调式
    
    @classmethod
    def get_western_equivalent(cls, mode: 'Mode') -> 'Mode':
        """获取最接近的西方调式等价"""
        western_map = {
            cls.CHINESE_GONG: cls.IONIAN,
            cls.CHINESE_SHANG: cls.MIXOLYDIAN,
            cls.CHINESE_JUE: cls.AEOLIAN,
            cls.CHINESE_ZHI: cls.LYDIAN,
            cls.CHINESE_YU: cls.DORIAN,
            cls.JAPANESE_IN: cls.PHRYGIAN,
            cls.JAPANESE_YO: cls.PENTATONIC_MAJOR,
            cls.MAQAM_RAST: cls.IONIAN,
            cls.MAQAM_BAYATI: cls.PHRYGIAN,
            cls.RAGA_BHAIRAV: cls.PHRYGIAN,
            cls.RAGA_PURVI: cls.LYDIAN
        }
        return western_map.get(mode, cls.IONIAN)  # 默认为伊奥尼亚(大调)

@dataclass
class Scale:
    """音阶定义"""
    root_note: str  # 根音(如"C"、"Eb")
    mode: Mode  # 调式
    intervals: List[float]  # 从根音起的半音数列表
    tuning_system: TuningSystem = TuningSystem.EQUAL_TEMPERAMENT
    
    @classmethod
    def create(cls, root: str, mode: Union[Mode, str], tuning: TuningSystem = TuningSystem.EQUAL_TEMPERAMENT) -> 'Scale':
        """创建指定根音和调式的音阶"""
        if isinstance(mode, str):
            try:
                mode = Mode[mode.upper()]
            except KeyError:
                # 尝试匹配mode名称
                for m in Mode:
                    if m.name.lower() == mode.lower():
                        mode = m
                        break
                else:
                    raise ValueError(f"未知调式: {mode}")
        
        # 定义各调式的音程结构
        intervals = {
            # 西方调式
            Mode.IONIAN: [0, 2, 4, 5, 7, 9, 11],  # 大调
            Mode.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            Mode.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
            Mode.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
            Mode.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            Mode.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],  # 小调
            Mode.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
            
            # 和声与旋律小调
            Mode.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
            Mode.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
            
            # 五声音阶
            Mode.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
            Mode.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
            
            # 印度拉格
            Mode.RAGA_BHAIRAV: [0, 1, 4, 5, 7, 8, 11],
            Mode.RAGA_TODI: [0, 1, 3, 6, 7, 8, 11],
            Mode.RAGA_PURVI: [0, 1, 4, 6, 7, 8, 11],
            
            # 阿拉伯马卡姆
            Mode.MAQAM_RAST: [0, 2, 3, 5, 7, 9, 10],
            Mode.MAQAM_BAYATI: [0, 1.5, 3, 5, 7, 8, 10],  # 包含微分音
            Mode.MAQAM_HIJAZ: [0, 1, 4, 5, 7, 8, 10],
            
            # 东亚调式
            Mode.CHINESE_GONG: [0, 2, 4, 7, 9],  # 宫调式(基于五声音阶)
            Mode.CHINESE_SHANG: [0, 2, 5, 7, 10],  # 商调式
            Mode.CHINESE_JUE: [0, 3, 5, 8, 10],  # 角调式
            Mode.CHINESE_ZHI: [0, 2, 5, 7, 9],  # 徵调式
            Mode.CHINESE_YU: [0, 3, 5, 7, 10],  # 羽调式
            
            Mode.JAPANESE_IN: [0, 1, 5, 7, 8],  # 日本阴调式
            Mode.JAPANESE_YO: [0, 2, 5, 7, 9],  # 日本阳调式
            
            # 非洲调式
            Mode.AFRICAN_KUMOI: [0, 2, 3, 7, 9],
            Mode.AFRICAN_BEBOP: [0, 2, 4, 5, 7, 9, 10, 11]
        }
        
        # 根据调律系统调整音程
        mode_intervals = intervals.get(mode, intervals[Mode.IONIAN])
        
        if tuning_system != TuningSystem.EQUAL_TEMPERAMENT:
            # 这里可以实现不同调律系统的精确音程计算
            # 例如纯律、中庸律等的实现
            pass
            
        return cls(
            root_note=root,
            mode=mode,
            intervals=mode_intervals,
            tuning_system=tuning_system
        )
    
    def get_notes(self) -> List[str]:
        """获取音阶的所有音符名称"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 根音索引
        if len(self.root_note) == 1:
            root_idx = note_names.index(self.root_note)
        else:
            # 处理带有升降号的根音
            note = self.root_note[0]
            accidental = self.root_note[1]
            root_idx = note_names.index(note)
            if accidental == '#':
                root_idx = (root_idx + 1) % 12
            elif accidental == 'b':
                root_idx = (root_idx - 1) % 12
        
        # 根据音程计算音符
        notes = []
        for interval in self.intervals:
            idx = (root_idx + int(interval)) % 12
            notes.append(note_names[idx])
            
        return notes

@dataclass
class RhythmPattern:
    """节奏模式定义"""
    name: str  # 节奏名称，如"4/4", "7/8"等
    beats_per_measure: int  # 每小节拍数
    beat_unit: int  # 以几分音符为一拍
    accent_pattern: List[float]  # 重音模式，值表示强度(0-1)
    subdivision_pattern: List[int] = field(default_factory=list)  # 细分模式
    
    @classmethod
    def create(cls, pattern_name: str) -> 'RhythmPattern':
        """从名称创建节奏模式"""
        # 解析常见节拍
        match = re.match(r'(\d+)/(\d+)', pattern_name)
        if not match:
            raise ValueError(f"无效的节拍格式: {pattern_name}")
            
        beats, unit = map(int, match.groups())
        
        # 为不同拍号定义默认重音模式
        if beats == 4 and unit == 4:  # 4/4
            accents = [1.0, 0.5, 0.8, 0.5]  # 强-弱-次强-弱
            subdivs = [2, 2, 2, 2]  # 每拍2个细分
        elif beats == 3 and unit == 4:  # 3/4
            accents = [1.0, 0.5, 0.5]  # 强-弱-弱
            subdivs = [2, 2, 2]
        elif beats == 6 and unit == 8:  # 6/8
            accents = [1.0, 0.3, 0.5, 0.7, 0.3, 0.5]  # 复合拍子
            subdivs = [1, 1, 1, 1, 1, 1]
        elif beats == 7 and unit == 8:  # 7/8
            accents = [1.0, 0.5, 0.5, 0.8, 0.4, 0.4, 0.4]  # 非对称拍子
            subdivs = [1, 1, 1, 1, 1, 1, 1]
        elif beats == 5 and unit == 4:  # 5/4
            accents = [1.0, 0.5, 0.8, 0.5, 0.7]  # 非对称拍子
            subdivs = [2, 2, 2, 2, 2]
        elif beats == 12 and unit == 8:  # 12/8
            accents = [1.0, 0.4, 0.4, 0.7, 0.4, 0.4, 0.8, 0.4, 0.4, 0.7, 0.4, 0.4]
            subdivs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            # 对于未预设的节拍，生成合理的重音模式
            accents = [1.0]  # 第一拍强
            for i in range(1, beats):
                # 每三拍一个次强拍
                if i % 3 == 0:
                    accents.append(0.8)
                else:
                    accents.append(0.5)
            subdivs = [2] * beats  # 默认每拍二等分
        
        return cls(
            name=pattern_name,
            beats_per_measure=beats,
            beat_unit=unit,
            accent_pattern=accents,
            subdivision_pattern=subdivs
        )
        
    def generate_beat_sequence(self, measures: int = 1) -> List[float]:
        """生成指定小节数的打击序列"""
        sequence = []
        for _ in range(measures):
            sequence.extend(self.accent_pattern)
        return sequence
    
    def generate_timing_grid(self, tempo: float, measures: int = 1) -> List[float]:
        """生成精确的时间网格(秒)"""
        beat_duration = 60 / tempo  # 一拍的秒数
        grid = []
        
        for _ in range(measures):
            measure_time = 0
            for i, subdivs in enumerate(self.subdivision_pattern):
                beat_time = measure_time
                for j in range(subdivs):
                    grid.append(beat_time + j * beat_duration / subdivs)
                measure_time += beat_duration
                
        return grid

class PerformanceTechnique(Enum):
    """演奏技巧"""
    # 通用技巧
    LEGATO = auto()           # 连奏
    STACCATO = auto()         # 断奏
    TREMOLO = auto()          # 颤音
    TRILL = auto()            # 颤音(装饰音)
    VIBRATO = auto()          # 揉音
    GLISSANDO = auto()        # 滑音
    PIZZICATO = auto()        # 拨弦
    
    # 文化特定技巧
    WESTERN_RUBATO = auto()           # 西方自由速度
    INDIAN_GAMAK = auto()             # 印度甘马克(精细音高装饰)
    CHINESE_PORTAMENTO = auto()       # 中国式滑音(细微平滑)
    AFRICAN_POLYRHYTHM = auto()       # 非洲多节奏
    FLAMENCO_RASGUEADO = auto()       # 弗拉门戈拨弦技巧
    MIDDLE_EASTERN_MICROTONES = auto()  # 中东微分音演奏
    
    @classmethod
    def get_techniques_for_culture(cls, culture: str) -> Set['PerformanceTechnique']:
        """获取特定文化常用的演奏技巧"""
        culture_techniques = {
            'Western': {cls.LEGATO, cls.STACCATO, cls.TRILL, cls.VIBRATO, cls.WESTERN_RUBATO},
            'Indian': {cls.LEGATO, cls.GLISSANDO, cls.VIBRATO, cls.INDIAN_GAMAK, cls.MIDDLE_EASTERN_MICROTONES},
            'Chinese': {cls.LEGATO, cls.TREMOLO, cls.CHINESE_PORTAMENTO, cls.VIBRATO},
            'African': {cls.STACCATO, cls.PIZZICATO, cls.AFRICAN_POLYRHYTHM},
            'Arabic': {cls.VIBRATO, cls.GLISSANDO, cls.MIDDLE_EASTERN_MICROTONES},
            'Japanese': {cls.LEGATO, cls.VIBRATO, cls.TREMOLO}
        }
        
        return culture_techniques.get(culture, {cls.LEGATO, cls.STACCATO})

@dataclass
class Instrument:
    """乐器定义"""
    name: str
    family: str
    pitch_range: Tuple[int, int]  # MIDI音符范围
    timbre_brightness: float  # 音色明亮度(0-1)
    attack_speed: float  # 起音速度(0-1)
    decay_time: float  # 衰减时间(0-1)
    sustain_ability: float  # 持续能力(0-1)
    techniques: Set[PerformanceTechnique]  # 支持的演奏技巧
    cultural_origin: str  # 文化起源
    
    @classmethod
    def create(cls, name: str, cultural_origin: str = None) -> 'Instrument':
        """创建预设乐器"""
        # 乐器数据库
        instruments_db = {
            # 西方乐器
            'Violin': {
                'family': 'Strings',
                'pitch_range': (55, 103),  # G3-G7
                'timbre_brightness': 0.7,
                'attack_speed': 0.8,
                'decay_time': 0.3,
                'sustain_ability': 0.9,
                'cultural_origin': 'Western'
            },
            'Piano': {
                'family': 'Keyboard',
                'pitch_range': (21, 108),  # A0-C8
                'timbre_brightness': 0.6,
                'attack_speed': 0.9,
                'decay_time': 0.7,
                'sustain_ability': 0.5,
                'cultural_origin': 'Western'
            },
            'Flute': {
                'family': 'Woodwind',
                'pitch_range': (60, 96),  # C4-C7
                'timbre_brightness': 0.8,
                'attack_speed': 0.7,
                'decay_time': 0.3,
                'sustain_ability': 0.8,
                'cultural_origin': 'Western'
            },
            'AcousticGuitar': {
                'family': 'Strings',
                'pitch_range': (40, 83),  # E2-B5
                'timbre_brightness': 0.6,
                'attack_speed': 0.7,
                'decay_time': 0.5,
                'sustain_ability': 0.6,
                'cultural_origin': 'Western'
            },
            
            # 印度乐器
            'Sitar': {
                'family': 'Strings',
                'pitch_range': (48, 84),  # C3-C6
                'timbre_brightness': 0.75,
                'attack_speed': 0.6,
                'decay_time': 0.8,
                'sustain_ability': 0.8,
                'cultural_origin': 'Indian'
            },
            'Tabla': {
                'family': 'Percussion',
                'pitch_range': (48, 72),  # C3-C5
                'timbre_brightness': 0.7,
                'attack_speed': 0.95,
                'decay_time': 0.4,
                'sustain_ability': 0.3,
                'cultural_origin': 'Indian'
            },
            
            # 中国乐器
            'Erhu': {
                'family': 'Strings',
                'pitch_range': (55, 91),  # G3-G6
                'timbre_brightness': 0.6,
                'attack_speed': 0.6,
                'decay_time': 0.4,
                'sustain_ability': 0.9,
                'cultural_origin': 'Chinese'
            },
            'Guzheng': {
                'family': 'Strings',
                'pitch_range': (48, 96),  # C3-C7
                'timbre_brightness': 0.7,
                'attack_speed': 0.8,
                'decay_time': 0.6,
                'sustain_ability': 0.5,
                'cultural_origin': 'Chinese'
            },
            
            # 非洲乐器
            'Djembe': {
                'family': 'Percussion',
                'pitch_range': (48, 60),  # C3-C4
                'timbre_brightness': 0.8,
                'attack_speed': 0.95,
                'decay_time': 0.3,
                'sustain_ability': 0.2,
                'cultural_origin': 'African'
            },
            'Kora': {
                'family': 'Strings',
                'pitch_range': (48, 84),  # C3-C6
                'timbre_brightness': 0.65,
                'attack_speed': 0.7,
                'decay_time': 0.6,
                'sustain_ability': 0.7,
                'cultural_origin': 'African'
            },
            
            # 电子/合成器
            'Synthesizer': {
                'family': 'Electronic',
                'pitch_range': (0, 127),  # 全范围
                'timbre_brightness': 0.8,
                'attack_speed': 0.9,
                'decay_time': 0.7,
                'sustain_ability': 1.0,
                'cultural_origin': 'Modern'
            }
        }
        
        if name not in instruments_db:
            raise ValueError(f"未知乐器: {name}")
        
        instrument_data = instruments_db[name]
        if cultural_origin:
            instrument_data['cultural_origin'] = cultural_origin
            
        # 获取适合该文化的演奏技巧
        techniques = PerformanceTechnique.get_techniques_for_culture(instrument_data['cultural_origin'])
        
        return cls(
            name=name,
            family=instrument_data['family'],
            pitch_range=instrument_data['pitch_range'],
            timbre_brightness=instrument_data['timbre_brightness'],
            attack_speed=instrument_data['attack_speed'],
            decay_time=instrument_data['decay_time'],
            sustain_ability=instrument_data['sustain_ability'],
            techniques=techniques,
            cultural_origin=instrument_data['cultural_origin']
        )

#####################################
# 文化音乐参数模型
#####################################

class CulturalElement(ABC):
    """文化元素基类"""
    @abstractmethod
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        pass

@dataclass
class MelodicCharacteristics(CulturalElement):
    """旋律特性"""
    interval_preference: List[float]  # 不同音程的使用频率
    ornament_density: float  # 装饰音密度
    phrase_structure: List[int]  # 乐句结构(每个乐句的小节数)
    contour_tendency: str  # 旋律轮廓趋势(ascending, descending, arching, etc)
    step_vs_leap_ratio: float  # 级进vs跳进比率(0-1)
    microtone_usage: float = 0.0  # 微分音使用强度(0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'interval_preference': self.interval_preference,
            'ornament_density': self.ornament_density,
            'phrase_structure': self.phrase_structure,
            'contour_tendency': self.contour_tendency,
            'step_vs_leap_ratio': self.step_vs_leap_ratio,
            'microtone_usage': self.microtone_usage
        }

#####################################
# 定义文化设置
#####################################
def get_culture_settings(culture):
    """
    根据选择的文化，返回相应的音乐设置。
    """
    culture_settings = {
        'Western': {
            'scales': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'rhythm_patterns': ['4/4', '3/4', '6/8'],
            'instruments': ['Violin', 'Piano', 'Synthesizer', 'Flute', 'AcousticGuitar', 'AcousticBass'],
            'temperature_scale': 1.0,
            'default_tonality': 'major',
            'alternative_tonality': 'minor',
            'note_density_scale': 1.0,
            'harmony_complexity': 1  # 1=基础和弦，2=七和弦，3=九和弦等
        },
        'Indian': {
            'scales': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'rhythm_patterns': ['7/8', '5/4', '9/8'],
            'instruments': ['Sitar', 'Tabla', 'Violin', 'Flute', 'Piano'],
            'temperature_scale': 1.2,
            'default_tonality': 'major',
            'alternative_tonality': 'minor',
            'note_density_scale': 1.1,
            'harmony_complexity': 2
        },
        'Chinese': {
            'scales': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'rhythm_patterns': ['4/4', '6/8', '5/4'],
            'instruments': ['Erhu', 'Guzheng', 'Flute', 'Piano', 'Synthesizer'],
            'temperature_scale': 1.0,
            'default_tonality': 'major',
            'alternative_tonality': 'minor',
            'note_density_scale': 1.0,
            'harmony_complexity': 1
        },
        'African': {
            'scales': ['C', 'Eb', 'F', 'G', 'Bb', 'C'],
            'rhythm_patterns': ['12/8', '6/8', '4/4'],
            'instruments': ['Djembe', 'Kora', 'Flute', 'Piano', 'Synthesizer'],
            'temperature_scale': 1.1,
            'default_tonality': 'major',
            'alternative_tonality': 'minor',
            'note_density_scale': 1.2,
            'harmony_complexity': 2
        }
    }
    
    return culture_settings[culture]

def get_advanced_culture_settings(culture):
    """
    获取高级文化设置，包括音乐理论模型中定义的详细参数。
    
    参数:
        culture (str): 文化名称，如'Western', 'Indian', 'Chinese', 'African'等
        
    返回:
        dict: 包含基本设置和高级设置的字典
    """
    # 首先获取基本设置
    basic_settings = get_culture_settings(culture)
    
    # 创建高级设置
    advanced_settings = {
        'Western': {
            'mode': Mode.IONIAN,
            'tuning_system': TuningSystem.EQUAL_TEMPERAMENT,
            'melodic_characteristics': MelodicCharacteristics(
                interval_preference=[0.3, 0.4, 0.1, 0.05, 0.05, 0.05, 0.02, 0.03],  # 偏好级进
                ornament_density=0.2,
                phrase_structure=[4, 4, 4, 4],  # 标准4小节乐句
                contour_tendency='arching',
                step_vs_leap_ratio=0.8,  # 大多是级进
                microtone_usage=0.0  # 无微分音
            ),
            'performance_techniques': list(PerformanceTechnique.get_techniques_for_culture('Western'))
        },
        'Indian': {
            'mode': Mode.RAGA_BHAIRAV,
            'tuning_system': TuningSystem.INDIAN_22_SHRUTI,
            'melodic_characteristics': MelodicCharacteristics(
                interval_preference=[0.2, 0.3, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05],
                ornament_density=0.7,  # 大量装饰音
                phrase_structure=[4, 6, 4, 6],  # 不规则乐句
                contour_tendency='descending',
                step_vs_leap_ratio=0.7,
                microtone_usage=0.4  # 使用微分音
            ),
            'performance_techniques': list(PerformanceTechnique.get_techniques_for_culture('Indian'))
        },
        'Chinese': {
            'mode': Mode.CHINESE_GONG,
            'tuning_system': TuningSystem.CHINESE_PENTATONIC,
            'melodic_characteristics': MelodicCharacteristics(
                interval_preference=[0.25, 0.35, 0.2, 0.1, 0.05, 0.03, 0.02],
                ornament_density=0.4,
                phrase_structure=[4, 4, 2, 2],
                contour_tendency='undulating',
                step_vs_leap_ratio=0.65,
                microtone_usage=0.1
            ),
            'performance_techniques': list(PerformanceTechnique.get_techniques_for_culture('Chinese'))
        },
        'African': {
            'mode': Mode.AFRICAN_KUMOI,
            'tuning_system': TuningSystem.AFRICAN_EQUIHEPTATONIC,
            'melodic_characteristics': MelodicCharacteristics(
                interval_preference=[0.3, 0.25, 0.2, 0.1, 0.1, 0.03, 0.02],
                ornament_density=0.3,
                phrase_structure=[2, 2, 2, 2, 2],  # 短小乐句
                contour_tendency='repetitive',
                step_vs_leap_ratio=0.7,
                microtone_usage=0.2
            ),
            'performance_techniques': list(PerformanceTechnique.get_techniques_for_culture('African'))
        }
    }
    
    # 检查是否有高级设置定义
    if culture not in advanced_settings:
        # 如果没有高级设置，创建默认设置
        advanced_settings[culture] = {
            'mode': Mode.IONIAN,
            'tuning_system': TuningSystem.EQUAL_TEMPERAMENT,
            'melodic_characteristics': MelodicCharacteristics(
                interval_preference=[0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02],
                ornament_density=0.2,
                phrase_structure=[4, 4],
                contour_tendency='arching',
                step_vs_leap_ratio=0.7,
                microtone_usage=0.0
            ),
            'performance_techniques': list(PerformanceTechnique.get_techniques_for_culture('Western'))
        }
    
    # 合并基本设置和高级设置
    result = basic_settings.copy()
    result.update({
        'advanced': advanced_settings[culture]
    })
    
    # 创建并添加实例化的对象
    root_note = "C"  # 默认根音为C
    mode = advanced_settings[culture]['mode']
    tuning = advanced_settings[culture]['tuning_system']
    
    # 添加音阶对象
    result['scale_object'] = Scale.create(root_note, mode, tuning)
    
    # 添加节奏模式对象
    if basic_settings['rhythm_patterns']:
        rhythm_patterns = []
        for pattern_name in basic_settings['rhythm_patterns']:
            try:
                rhythm_pattern = RhythmPattern.create(pattern_name)
                rhythm_patterns.append(rhythm_pattern)
            except ValueError as e:
                print(f"警告: 无法创建节奏模式 {pattern_name}: {e}")
        result['rhythm_pattern_objects'] = rhythm_patterns
    
    # 添加乐器对象
    if basic_settings['instruments']:
        instruments = []
        for instr_name in basic_settings['instruments']:
            try:
                instrument = Instrument.create(instr_name, culture)
                instruments.append(instrument)
            except ValueError as e:
                print(f"警告: 无法创建乐器 {instr_name}: {e}")
        result['instrument_objects'] = instruments
    
    return result