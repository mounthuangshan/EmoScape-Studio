import os
import numpy as np
import argparse
from scipy.io import wavfile
import json
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
import time

# 音乐理论数据
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],  # 大调
    "minor": [0, 2, 3, 5, 7, 8, 10],  # 小调
    "pentatonic": [0, 2, 4, 7, 9],    # 五声音阶
    "blues": [0, 3, 5, 6, 7, 10],     # 布鲁斯音阶
    "dorian": [0, 2, 3, 5, 7, 9, 10], # 多利亚调式
    "phrygian": [0, 1, 3, 5, 7, 8, 10], # 弗里几亚调式
    "lydian": [0, 2, 4, 6, 7, 9, 11], # 利底亚调式
    "mixolydian": [0, 2, 4, 5, 7, 9, 10], # 混合利底亚调式
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11], # 减音阶
    "locrian": [0, 1, 3, 5, 6, 8, 10], # 洛克里亚调式
}

# 基础音符频率 (A4 = 440Hz)
A4_FREQ = 440.0
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 计算频率
def get_frequency(note, octave):
    """计算特定音符和八度的频率"""
    note_index = NOTES.index(note)
    # A4是第49个键，索引从0开始，所以A4的索引是48
    semitones_from_a4 = (octave - 4) * 12 + note_index - 9
    return A4_FREQ * (2 ** (semitones_from_a4 / 12))

# 基于情感参数调整音乐生成
def get_emotion_music_params(emotion, valence=0.5, arousal=0.5, intensity=0.8):
    """根据情感类型和参数获取音乐生成参数"""
    params = {
        "tempo": 100,              # BPM
        "scale": "major",          # 音阶
        "base_note": "C",          # 基础音符
        "octave": 4,               # 八度
        "rhythm_complexity": 0.5,  # 节奏复杂度
        "harmony_density": 0.5,    # 和声密度
        "note_duration": 0.25,     # 音符持续时间(秒)
        "reverb": 0.3,             # 混响量
        "timbre_brightness": 0.5,  # 音色明亮度
    }
    
    # 基于情感类型调整参数
    if emotion in ["joy", "anticipation", "trust"]:
        # 积极情感: 明亮的大调，较快的节奏
        params["scale"] = "major"
        params["tempo"] = 110 + random.randint(0, 30)
        params["timbre_brightness"] = 0.7 + random.random() * 0.3
        params["rhythm_complexity"] = 0.6 + random.random() * 0.3
    
    elif emotion in ["sadness", "disgust"]:
        # 消极情感: 暗淡的小调，较慢的节奏
        params["scale"] = "minor"
        params["tempo"] = 60 + random.randint(0, 20)
        params["timbre_brightness"] = 0.2 + random.random() * 0.3
        params["note_duration"] = 0.4 + random.random() * 0.3
        params["reverb"] = 0.6 + random.random() * 0.3
    
    elif emotion in ["fear", "anger"]:
        # 紧张情感: 不协和的音阶，强烈的节奏
        params["scale"] = random.choice(["diminished", "locrian", "phrygian"])
        params["tempo"] = 100 + random.randint(0, 40)
        params["rhythm_complexity"] = 0.7 + random.random() * 0.3
        params["harmony_density"] = 0.7 + random.random() * 0.3
    
    elif emotion in ["surprise"]:
        # 惊奇情感: 明亮但不寻常的音阶，变化的节奏
        params["scale"] = random.choice(["lydian", "mixolydian"])
        params["tempo"] = 90 + random.randint(0, 50)
        params["rhythm_complexity"] = 0.8 + random.random() * 0.2
    
    # 进一步基于效价和唤醒度调整参数
    # 效价影响调式和音色
    if valence > 0.7:
        if params["scale"] == "minor":
            params["scale"] = "major"
        params["timbre_brightness"] += 0.2
    elif valence < 0.3:
        if params["scale"] == "major":
            params["scale"] = "minor"
        params["timbre_brightness"] -= 0.2
    
    # 唤醒度影响速度和节奏复杂度
    if arousal > 0.7:
        params["tempo"] += 20
        params["rhythm_complexity"] += 0.2
        params["note_duration"] *= 0.8
    elif arousal < 0.3:
        params["tempo"] -= 20
        params["rhythm_complexity"] -= 0.2
        params["note_duration"] *= 1.3
        params["reverb"] += 0.2
    
    # 强度影响整体表现力
    params["volume"] = 0.5 + intensity * 0.5
    
    # 确保参数在合理范围内
    params["tempo"] = max(40, min(220, params["tempo"]))
    params["timbre_brightness"] = max(0.1, min(1.0, params["timbre_brightness"]))
    params["rhythm_complexity"] = max(0.1, min(1.0, params["rhythm_complexity"]))
    params["harmony_density"] = max(0.1, min(1.0, params["harmony_density"]))
    params["note_duration"] = max(0.1, min(1.0, params["note_duration"]))
    params["reverb"] = max(0.0, min(0.9, params["reverb"]))
    
    return params

# 生成简单音频波形
def generate_tone(frequency, duration, volume=1.0, sample_rate=44100, fade=0.1):
    """生成单一音调"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 创建基础正弦波
    tone = np.sin(2 * np.pi * frequency * t) * volume
    
    # 添加泛音以丰富音色
    for i in range(2, 6):
        tone += np.sin(2 * np.pi * frequency * i * t) * volume / (i * 2)
    
    # 应用淡入淡出
    fade_samples = int(fade * sample_rate)
    if fade_samples * 2 < len(tone):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    
    return tone

def apply_envelope(audio, attack_time=0.01, decay_time=0.1, sustain_level=0.7, release_time=0.3, sample_rate=44100):
    """应用ADSR包络到音频"""
    attack = int(attack_time * sample_rate)
    decay = int(decay_time * sample_rate)
    release = int(release_time * sample_rate)
    
    total_samples = len(audio)
    envelope = np.ones(total_samples)
    
    # Attack阶段
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    
    # Decay阶段
    if decay > 0:
        decay_end = min(attack + decay, total_samples)
        decay_samples = decay_end - attack
        if decay_samples > 0:
            envelope[attack:decay_end] = np.linspace(1, sustain_level, decay_samples)
    
    # Sustain阶段已经默认设置为sustain_level
    
    # Release阶段
    if release > 0:
        release_start = max(0, total_samples - release)
        release_samples = total_samples - release_start
        if release_samples > 0:
            envelope[release_start:] = np.linspace(envelope[release_start], 0, release_samples)
    
    return audio * envelope

def apply_reverb(audio, reverb_amount=0.3, delay=0.1, decay=0.5, sample_rate=44100):
    """应用简单的混响效果"""
    if reverb_amount <= 0:
        return audio
    
    delay_samples = int(delay * sample_rate)
    if delay_samples <= 0:
        return audio
    
    # 创建延迟音频
    delayed_audio = np.zeros_like(audio)
    delayed_audio[delay_samples:] = audio[:-delay_samples] * decay * reverb_amount
    
    # 混合原始和延迟音频
    result = audio + delayed_audio
    
    # 标准化以避免削波
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result))
    
    return result

class MelodyGenerator:
    """旋律生成器"""
    def __init__(self, params):
        self.params = params
        self.scale = SCALES[params["scale"]]
        self.base_freq = get_frequency(params["base_note"], params["octave"])
        self.tempo = params["tempo"]
        self.note_duration = params["note_duration"]
        self.sample_rate = 44100
    
    def generate(self, duration=10.0, collab_mode="standalone"):
        """生成旋律"""
        # 根据BPM和音符持续时间计算音符数量
        beat_duration = 60.0 / self.tempo  # 一拍的时长（秒）
        note_beats = self.note_duration / beat_duration  # 每个音符占多少拍
        total_notes = int(duration / (beat_duration * note_beats))
        
        # 生成随机音符序列
        notes = []
        prev_note_idx = random.randint(0, len(self.scale) - 1)
        
        for i in range(total_notes):
            # 根据协作模式调整生成策略
            if collab_mode == "structure_focused":
                # 分层生成模式：更强的结构感，按照音乐句子组织
                if i % 8 == 0:  # 每8个音符一个乐句
                    # 乐句开始，倾向于用根音或五音
                    step = random.choice([0, 4])
                    prev_note_idx = step % len(self.scale)
                elif i % 8 == 7:  # 乐句结束
                    # 乐句结束，倾向于解决到稳定音
                    step = random.choice([0, 4, 6])
                    prev_note_idx = step % len(self.scale)
                else:
                    # 乐句中间，较小的音程变化
                    step_change = random.choice([-2, -1, 0, 1, 2])
                    prev_note_idx = (prev_note_idx + step_change) % len(self.scale)
            
            elif collab_mode == "emotion_focused":
                # 混合生成模式：更专注于情感表达
                complexity = self.params["rhythm_complexity"]
                if random.random() < complexity:
                    # 复杂变化，大跳
                    step_change = random.choice([-4, -3, 3, 4])
                else:
                    # 平滑变化，小跳
                    step_change = random.choice([-1, 0, 1])
                prev_note_idx = (prev_note_idx + step_change) % len(self.scale)
            
            else:  # 默认生成模式
                # 简单的步进逻辑
                step_change = random.choice([-2, -1, 0, 1, 2])
                prev_note_idx = (prev_note_idx + step_change) % len(self.scale)
            
            # 获取音阶中的音符
            scale_note = self.scale[prev_note_idx]
            
            # 决定八度（随机增加或减少八度的概率）
            octave_shift = 0
            if random.random() < 0.1:
                octave_shift = random.choice([-1, 1])
            
            # 计算最终音符频率
            note_freq = self.base_freq * (2 ** (scale_note / 12)) * (2 ** octave_shift)
            notes.append(note_freq)
        
        # 生成音频
        audio_data = np.zeros(int(duration * self.sample_rate))
        time_position = 0
        
        for note_freq in notes:
            # 实际持续时间稍微短于理论持续时间，留出间隙
            actual_duration = beat_duration * note_beats * 0.95
            
            # 随机调整音符长度，增加自然感
            variation = 1.0 + (random.random() * 0.1 - 0.05)
            note_duration = actual_duration * variation
            
            # 生成音符
            note_samples = int(note_duration * self.sample_rate)
            note_audio = generate_tone(
                note_freq, 
                note_duration, 
                volume=self.params["volume"],
                sample_rate=self.sample_rate
            )
            
            # 应用音符包络
            note_audio = apply_envelope(
                note_audio,
                attack_time=0.01,
                decay_time=0.1,
                sustain_level=0.7,
                release_time=0.2
            )
            
            # 添加到主音频流
            end_pos = min(time_position + len(note_audio), len(audio_data))
            audio_data[time_position:end_pos] += note_audio[:end_pos-time_position]
            
            # 移动时间位置
            time_position += int(beat_duration * note_beats * self.sample_rate)
            if time_position >= len(audio_data):
                break
        
        # 应用效果
        audio_data = apply_reverb(
            audio_data, 
            reverb_amount=self.params["reverb"],
            delay=0.1,
            decay=0.5
        )
        
        # 标准化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data

class HarmonyGenerator:
    """和声生成器"""
    def __init__(self, params):
        self.params = params
        self.scale = SCALES[params["scale"]]
        self.base_freq = get_frequency(params["base_note"], params["octave"]-1)  # 低一个八度
        self.tempo = params["tempo"]
        self.note_duration = params["note_duration"] * 4  # 和声持续时间更长
        self.sample_rate = 44100
        self.density = params["harmony_density"]
    
    def generate(self, duration=10.0, melody=None, collab_mode="standalone"):
        """生成和声"""
        # 计算每个和弦的持续时间
        beat_duration = 60.0 / self.tempo
        chord_duration = self.note_duration
        chords_count = int(duration / chord_duration)
        
        # 基本和弦进行
        chord_progression = []
        
        # 根据协作模式调整和弦生成
        if collab_mode == "structure_focused":
            # 分层模式：使用常见和弦进行
            if self.params["scale"] == "major":
                # 大调常见和弦进行
                progressions = [
                    [0, 5, 3, 4],  # I-VI-IV-V
                    [0, 3, 4, 0],  # I-IV-V-I
                    [0, 5, 1, 4],  # I-VI-II-V
                    [0, 3, 0, 4]   # I-IV-I-V
                ]
            else:
                # 小调常见和弦进行
                progressions = [
                    [0, 5, 3, 4],  # i-VI-iv-V
                    [0, 5, 3, 0],  # i-VI-iv-i
                    [0, 3, 4, 0],  # i-iv-V-i
                    [0, 5, 0, 4]   # i-VI-i-V
                ]
            
            # 选择一个和弦进行
            base_progression = random.choice(progressions)
            
            # 重复和弦进行直到达到所需时长
            while len(chord_progression) < chords_count:
                chord_progression.extend(base_progression)
            
            # 截断多余部分
            chord_progression = chord_progression[:chords_count]
        
        elif collab_mode == "emotion_focused":
            # 混合模式：根据情感参数选择和弦
            if self.params["timbre_brightness"] > 0.6:
                # 明亮情感，使用大三和弦
                chord_types = ["maj", "maj7", "6"]
            elif self.params["timbre_brightness"] < 0.4:
                # 暗淡情感，使用小三和弦和减七和弦
                chord_types = ["min", "min7", "dim7"]
            else:
                # 中性情感，混合和弦类型
                chord_types = ["maj", "min", "7", "maj7"]
            
            # 随机生成和弦进行
            prev_chord = 0
            for i in range(chords_count):
                # 倾向于选择和声功能上相关的和弦
                weights = [0.2] * len(self.scale)
                
                # 增强主要和弦的权重
                weights[0] = 0.4  # 主和弦
                if len(weights) > 4:
                    weights[4] = 0.3  # 属和弦
                if len(weights) > 3:
                    weights[3] = 0.3  # 下属和弦
                
                # 归一化权重
                weights = [w/sum(weights) for w in weights]
                
                # 选择和弦
                chord = random.choices(range(len(self.scale)), weights=weights)[0]
                chord_progression.append(chord)
                prev_chord = chord
        
        else:
            # 默认模式：简单随机和弦
            for i in range(chords_count):
                chord = random.randint(0, len(self.scale) - 1)
                chord_progression.append(chord)
        
        # 生成音频
        audio_data = np.zeros(int(duration * self.sample_rate))
        
        for i, chord_root in enumerate(chord_progression):
            # 计算和弦音符（三和弦：根音、三音、五音）
            chord_notes = []
            chord_root_idx = chord_root % len(self.scale)
            
            # 根音
            root_note = self.scale[chord_root_idx]
            chord_notes.append(root_note)
            
            # 三音 (根音上方2个音阶音符)
            third_idx = (chord_root_idx + 2) % len(self.scale)
            third_note = self.scale[third_idx]
            if third_note < root_note:
                third_note += 12  # 确保是高于根音的
            chord_notes.append(third_note)
            
            # 五音 (根音上方4个音阶音符)
            fifth_idx = (chord_root_idx + 4) % len(self.scale)
            fifth_note = self.scale[fifth_idx]
            if fifth_note < root_note:
                fifth_note += 12  # 确保是高于根音的
            chord_notes.append(fifth_note)
            
            # 七音 (根据密度可能添加)
            if random.random() < self.density:
                seventh_idx = (chord_root_idx + 6) % len(self.scale)
                seventh_note = self.scale[seventh_idx]
                if seventh_note < root_note:
                    seventh_note += 12
                chord_notes.append(seventh_note)
            
            # 生成每个和弦音符的音频并混合
            chord_audio = np.zeros(int(chord_duration * self.sample_rate))
            
            for note in chord_notes:
                # 计算频率
                freq = self.base_freq * (2 ** (note / 12))
                
                # 生成音符
                note_audio = generate_tone(
                    freq, 
                    chord_duration,
                    volume=self.params["volume"] * 0.6,  # 和声音量稍低
                    sample_rate=self.sample_rate
                )
                
                # 应用包络
                note_audio = apply_envelope(
                    note_audio,
                    attack_time=0.05,
                    decay_time=0.2,
                    sustain_level=0.6,
                    release_time=0.5
                )
                
                # 添加到和弦
                chord_audio += note_audio
            
            # 确保和弦不会过载
            if np.max(np.abs(chord_audio)) > 0:
                chord_audio = chord_audio / np.max(np.abs(chord_audio))
            
            # 将和弦添加到主音频
            start_pos = int(i * chord_duration * self.sample_rate)
            end_pos = int((i + 1) * chord_duration * self.sample_rate)
            end_pos = min(end_pos, len(audio_data))
            audio_data[start_pos:end_pos] += chord_audio[:end_pos-start_pos]
        
        # 应用效果
        audio_data = apply_reverb(
            audio_data, 
            reverb_amount=self.params["reverb"] * 1.5,  # 和声混响更多
            delay=0.15,
            decay=0.6
        )
        
        # 标准化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.7  # 和声音量降低
        
        return audio_data

class PercussionGenerator:
    """打击乐生成器"""
    def __init__(self, params):
        self.params = params
        self.tempo = params["tempo"]
        self.complexity = params["rhythm_complexity"]
        self.sample_rate = 44100
    
    def generate_kick(self, duration=0.2):
        """生成低音鼓声音"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # 频率从高快速降低
        freq = 150 * np.exp(-t * 20)
        
        # 创建基础正弦波
        audio = np.sin(2 * np.pi * np.cumsum(freq) / self.sample_rate)
        
        # 应用包络
        envelope = np.exp(-t * 20)
        audio = audio * envelope
        
        return audio
    
    def generate_snare(self, duration=0.2):
        """生成军鼓声音"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # 白噪声
        noise = np.random.randn(len(t))
        
        # 加上一些中频正弦波
        tone = np.sin(2 * np.pi * 180 * t) * 0.5
        
        # 混合并应用包络
        audio = (noise * 0.7 + tone * 0.3) * np.exp(-t * 15)
        
        return audio
    
    def generate_hihat(self, duration=0.1, closed=True):
        """生成高帽声音"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # 白噪声
        audio = np.random.randn(len(t))
        
        # 应用高通滤波（简化版）
        for i in range(1, len(audio)):
            audio[i] = 0.9 * audio[i] + 0.1 * audio[i-1]
        
        # 应用包络
        decay = 50 if closed else 10
        envelope = np.exp(-t * decay)
        audio = audio * envelope
        
        return audio
    
    def generate(self, duration=10.0, collab_mode="standalone"):
        """生成打击乐"""
        beat_duration = 60.0 / self.tempo
        total_beats = int(duration / beat_duration)
        
        # 创建空音频
        audio_data = np.zeros(int(duration * self.sample_rate))
        
        # 根据协作模式选择节奏模式
        if collab_mode == "structure_focused":
            # 分层模式：规则的节奏模式
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0]  # 4/4拍，1和5拍有低音鼓
            snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0]  # 3和7拍有军鼓
            hihat_pattern = [1, 1, 1, 1, 1, 1, 1, 1]  # 每拍都有高帽
            
            # 根据复杂度可能添加变化
            if random.random() < self.complexity:
                # 添加鼓点变化
                if random.random() < 0.5:
                    kick_pattern[2] = 0.7  # 3拍弱低音鼓
                if random.random() < 0.4:
                    kick_pattern[6] = 0.8  # 7拍弱低音鼓
                if random.random() < 0.3:
                    snare_pattern[1] = 0.5  # 2拍弱军鼓
        
        elif collab_mode == "emotion_focused":
            # 混合模式：基于情感的节奏
            if self.params["tempo"] > 120:
                # 快节奏，紧张感
                kick_pattern = [1, 0, 0.8, 0, 1, 0, 0.6, 0]
                snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0.7]
                hihat_pattern = [1, 0.8, 1, 0.8, 1, 0.8, 1, 0.8]
            elif self.params["tempo"] < 80:
                # 慢节奏，沉稳感
                kick_pattern = [1, 0, 0, 0, 0.8, 0, 0, 0]
                snare_pattern = [0, 0, 0.7, 0, 0, 0, 1, 0]
                hihat_pattern = [0.8, 0, 0.9, 0, 0.8, 0, 0.9, 0]
            else:
                # 中等节奏
                kick_pattern = [1, 0, 0, 0.5, 1, 0, 0, 0]
                snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0]
                hihat_pattern = [1, 0.7, 1, 0.7, 1, 0.7, 1, 0.7]
        
        else:
            # 默认模式：基本节奏
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0]
            snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0]
            hihat_pattern = [1, 0, 1, 0, 1, 0, 1, 0]
            
            # 根据复杂度随机添加打击乐
            if self.complexity > 0.6:
                hihat_pattern = [1, 0.7, 1, 0.7, 1, 0.7, 1, 0.7]
        
        # 生成打击乐音频
        for beat in range(total_beats):
            beat_idx = beat % len(kick_pattern)
            
            # 低音鼓
            if kick_pattern[beat_idx] > 0:
                kick = self.generate_kick() * kick_pattern[beat_idx] * self.params["volume"]
                start_pos = int(beat * beat_duration * self.sample_rate)
                end_pos = min(start_pos + len(kick), len(audio_data))
                audio_data[start_pos:end_pos] += kick[:end_pos-start_pos]
            
            # 军鼓
            if snare_pattern[beat_idx] > 0:
                snare = self.generate_snare() * snare_pattern[beat_idx] * self.params["volume"]
                start_pos = int(beat * beat_duration * self.sample_rate)
                end_pos = min(start_pos + len(snare), len(audio_data))
                audio_data[start_pos:end_pos] += snare[:end_pos-start_pos]
            
            # 高帽
            if hihat_pattern[beat_idx] > 0:
                # 交替闭合/开放高帽
                closed = beat_idx % 2 == 0
                hihat = self.generate_hihat(closed=closed) * hihat_pattern[beat_idx] * self.params["volume"] * 0.7
                start_pos = int(beat * beat_duration * self.sample_rate)
                end_pos = min(start_pos + len(hihat), len(audio_data))
                audio_data[start_pos:end_pos] += hihat[:end_pos-start_pos]
            
            # 基于复杂度添加随机鼓点变化
            if random.random() < self.complexity * 0.3:
                # 添加额外的高帽
                extra_pos = int((beat + 0.5) * beat_duration * self.sample_rate)
                if extra_pos + 1000 < len(audio_data):
                    hihat = self.generate_hihat(duration=0.05) * 0.5 * self.params["volume"]
                    end_pos = min(extra_pos + len(hihat), len(audio_data))
                    audio_data[extra_pos:end_pos] += hihat[:end_pos-extra_pos]
        
        # 标准化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        return audio_data

class AmbientGenerator:
    """环境音效生成器"""
    def __init__(self, params):
        self.params = params
        self.scale = SCALES[params["scale"]]
        self.base_freq = get_frequency(params["base_note"], params["octave"])
        self.reverb = params["reverb"]
        self.brightness = params["timbre_brightness"]
        self.sample_rate = 44100
    
    def generate(self, duration=10.0, collab_mode="standalone"):
        """生成环境音效"""
        audio_data = np.zeros(int(duration * self.sample_rate))
        
        # 根据协作模式选择不同的环境音效生成策略
        if collab_mode == "structure_focused":
            # 分层模式：更加均匀，有结构的环境音效
            
            # 生成低频、持续时间长的声音层
            pad_freq = self.base_freq * 0.5
            pad_audio = np.zeros(len(audio_data))
            
            # 缓慢变化的LFO调制
            t = np.linspace(0, duration, len(audio_data))
            lfo1 = np.sin(2 * np.pi * 0.1 * t) * 0.1
            lfo2 = np.sin(2 * np.pi * 0.05 * t) * 0.2
            
            # 应用调制
            for i in range(0, len(audio_data), self.sample_rate // 10):
                chunk_size = min(self.sample_rate // 10, len(audio_data) - i)
                t_chunk = np.linspace(0, chunk_size / self.sample_rate, chunk_size)
                
                # 调制频率
                mod_freq = pad_freq * (1 + lfo1[i])
                
                # 生成音频片段
                pad_chunk = np.sin(2 * np.pi * mod_freq * t_chunk)
                
                # 调制音量
                volume_mod = 0.3 + lfo2[i]
                pad_chunk = pad_chunk * volume_mod
                
                # 添加到主pad
                pad_audio[i:i+chunk_size] += pad_chunk
            
            # 应用低通滤波（简单实现）
            filtered_pad = np.zeros_like(pad_audio)
            filtered_pad[0] = pad_audio[0]
            alpha = 0.9  # 滤波系数
            for i in range(1, len(pad_audio)):
                filtered_pad[i] = alpha * filtered_pad[i-1] + (1 - alpha) * pad_audio[i]
            
            # 添加到主音频
            audio_data += filtered_pad * self.params["volume"] * 0.4
        
        elif collab_mode == "emotion_focused":
            # 混合模式：情感主导的环境音效
            
            # 根据亮度参数选择音色特性
            if self.brightness > 0.6:
                # 明亮音色
                harmonics = [1, 0.5, 0.3, 0.4, 0.2]
                filter_alpha = 0.6
            elif self.brightness < 0.4:
                # 暗沉音色
                harmonics = [1, 0.2, 0.1, 0.05, 0.02]
                filter_alpha = 0.95
            else:
                # 中性音色
                harmonics = [1, 0.3, 0.2, 0.1, 0.05]
                filter_alpha = 0.8
            
            # 生成几个音符形成环境音效
            for note_idx in random.sample(range(len(self.scale)), 3):
                note = self.scale[note_idx]
                freq = self.base_freq * (2 ** (note / 12))
                
                # 为每个音符创建缓慢演变的音频
                t = np.linspace(0, duration, len(audio_data))
                note_audio = np.zeros_like(audio_data)
                
                # 添加谐波
                for i, harmonic_vol in enumerate(harmonics):
                    harmonic = np.sin(2 * np.pi * freq * (i + 1) * t) * harmonic_vol
                    note_audio += harmonic
                
                # 应用包络
                env = np.ones_like(audio_data)
                attack = int(0.1 * self.sample_rate)
                release = int(0.2 * self.sample_rate)
                env[:attack] = np.linspace(0, 1, attack)
                env[-release:] = np.linspace(1, 0, release)
                
                # 应用缓慢的音量变化
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
                env = env * mod
                
                note_audio = note_audio * env * self.params["volume"] * 0.2
                
                # 添加到主音频
                audio_data += note_audio
        
        else:
            # 默认模式：简单的基于噪声的环境音效
            
            # 创建白噪声
            noise = np.random.randn(len(audio_data)) * 0.1
            
            # 应用滤波
            filtered_noise = np.zeros_like(noise)
            filtered_noise[0] = noise[0]
            alpha = 0.95  # 滤波系数
            for i in range(1, len(noise)):
                filtered_noise[i] = alpha * filtered_noise[i-1] + (1 - alpha) * noise[i]
            
            # 添加缓慢变化的音量包络
            t = np.linspace(0, duration, len(audio_data))
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
            
            audio_data += filtered_noise * env * self.params["volume"] * 0.3
        
        # 应用大量混响
        audio_data = apply_reverb(
            audio_data, 
            reverb_amount=self.reverb * 2.0,  # 环境音效需要更多混响
            delay=0.2,
            decay=0.8
        )
        
        # 标准化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.5  # 环境声音应该更安静
        
        return audio_data

class MusicGenerator:
    """音乐生成器组合类"""
    def __init__(self, emotion="joy", valence=0.7, arousal=0.6, intensity=0.8):
        self.emotion = emotion
        self.valence = valence
        self.arousal = arousal
        self.intensity = intensity
        
        # 获取音乐参数
        self.params = get_emotion_music_params(emotion, valence, arousal, intensity)
        
        # 初始化各个生成器
        self.melody_gen = MelodyGenerator(self.params)
        self.harmony_gen = HarmonyGenerator(self.params)
        self.percussion_gen = PercussionGenerator(self.params)
        self.ambient_gen = AmbientGenerator(self.params)
        
        self.sample_rate = 44100
    
    def _generate_instruments_track(self, duration=10.0, collab_mode="standalone"):
        """生成乐器特性轨道，提供更突出的音色特征"""
        # 创建一个新的参数集，基于当前参数但调整了一些特性
        instrument_params = self.params.copy()
        
        # 根据情感类型选择不同的乐器特性
        if self.valence > 0.7:  # 高效价（积极情绪）
            # 明亮音色，高八度
            instrument_params["octave"] = self.params["octave"] + 1
            instrument_params["timbre_brightness"] = min(1.0, self.params["timbre_brightness"] * 1.3)
            instrument_params["note_duration"] = self.params["note_duration"] * 0.8  # 更短促的音符
        elif self.valence < 0.3:  # 低效价（消极情绪）
            # 暗淡音色，低八度
            instrument_params["octave"] = max(1, self.params["octave"] - 1)
            instrument_params["reverb"] = min(0.9, self.params["reverb"] * 1.5)  # 更多混响
            instrument_params["note_duration"] = self.params["note_duration"] * 1.3  # 更持久的音符
        
        # 创建旋律和和声生成器的混合音轨
        melody_gen = MelodyGenerator(instrument_params)
        harmony_gen = HarmonyGenerator(instrument_params)
        
        # 生成基础音频
        if collab_mode == "emotion_focused":
            melody_part = melody_gen.generate(duration, collab_mode=collab_mode)
            harmony_part = harmony_gen.generate(duration, collab_mode=collab_mode)
        else:
            melody_part = melody_gen.generate(duration)
            harmony_part = harmony_gen.generate(duration)
        
        # 混合音轨，调整混合比例突出乐器特性
        if self.arousal > 0.7:  # 高唤醒度
            # 旋律部分更突出
            instruments_track = melody_part * 0.7 + harmony_part * 0.3
        elif self.arousal < 0.3:  # 低唤醒度
            # 和声部分更突出
            instruments_track = melody_part * 0.3 + harmony_part * 0.7
        else:
            # 平衡混合
            instruments_track = melody_part * 0.5 + harmony_part * 0.5
        
        # 标准化
        if np.max(np.abs(instruments_track)) > 0:
            instruments_track = instruments_track / np.max(np.abs(instruments_track)) * 0.8
        
        return instruments_track
    
    def generate_parallel(self, duration=10.0):
        """并行生成模式 - 各生成器独立工作"""
        # 各自独立生成
        melody = self.melody_gen.generate(duration, collab_mode="parallel")
        harmony = self.harmony_gen.generate(duration, collab_mode="parallel")
        percussion = self.percussion_gen.generate(duration, collab_mode="parallel")
        ambient = self.ambient_gen.generate(duration, collab_mode="parallel")
        # 添加乐器轨道
        instruments = self._generate_instruments_track(duration, collab_mode="parallel")
        
        # 混合各轨道
        result = (
            melody * 0.7 + 
            harmony * 0.5 + 
            percussion * 0.6 + 
            ambient * 0.4 +
            instruments * 0.5
        )
        
        # 标准化
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return {
            "combined": result,
            "melody": melody,
            "harmony": harmony, 
            "percussion": percussion,
            "ambient": ambient,
            "instruments": instruments  # 添加乐器轨道
        }
    
    def generate_pipeline(self, duration=10.0):
        """分层生成模式 - 按顺序生成，前一阶段的输出影响后一阶段"""
        # 1. 首先生成环境音效作为基础
        ambient = self.ambient_gen.generate(duration, collab_mode="structure_focused")
        
        # 2. 基于环境音效生成和声
        harmony = self.harmony_gen.generate(duration, collab_mode="structure_focused")
        
        # 3. 基于和声生成旋律
        melody = self.melody_gen.generate(duration, collab_mode="structure_focused")
        
        # 4. 最后加入打击乐
        percussion = self.percussion_gen.generate(duration, collab_mode="structure_focused")
        
        instruments = self._generate_instruments_track(duration, collab_mode="structure_focused")
        
        # 修改混合部分包含instruments
        result = (
            melody * 0.7 + 
            harmony * 0.5 + 
            percussion * 0.6 + 
            ambient * 0.4 +
            instruments * 0.5
        )
        
        # 标准化
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return {
            "combined": result,
            "melody": melody,
            "harmony": harmony, 
            "percussion": percussion,
            "ambient": ambient,
            "instruments": instruments
        }
    
    def generate_hybrid(self, duration=10.0):
        """混合生成模式 - 情感驱动的多样化音乐"""
        # 1. 使用情感参数生成所有轨道
        melody = self.melody_gen.generate(duration, collab_mode="emotion_focused")
        harmony = self.harmony_gen.generate(duration, collab_mode="emotion_focused")
        percussion = self.percussion_gen.generate(duration, collab_mode="emotion_focused")
        ambient = self.ambient_gen.generate(duration, collab_mode="emotion_focused")
        
        # 根据情感参数调整混合比例
        melody_weight = 0.7
        harmony_weight = 0.5
        percussion_weight = 0.6
        ambient_weight = 0.4
        
        # 根据唤醒度调整打击乐比例
        percussion_weight *= (0.5 + self.arousal * 0.5)
        
        # 根据效价调整和声和环境音比例
        if self.valence < 0.4:  # 负面情感
            harmony_weight *= 1.2
            ambient_weight *= 1.3
            melody_weight *= 0.8
        
        instruments = self._generate_instruments_track(duration, collab_mode="emotion_focused")
        
        # 添加乐器权重
        instruments_weight = 0.5
        
        # 根据情感调整乐器权重
        if self.arousal > 0.7:
            instruments_weight *= 1.2
        
        # 混合各轨道
        result = (
            melody * melody_weight + 
            harmony * harmony_weight + 
            percussion * percussion_weight + 
            ambient * ambient_weight +
            instruments * instruments_weight
        )
        
        # 标准化
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return {
            "combined": result,
            "melody": melody,
            "harmony": harmony, 
            "percussion": percussion,
            "ambient": ambient,
            "instruments": instruments
        }
    
    def generate_rl(self, duration=10.0):
        """强化学习协作模式 - 模拟自适应音乐生成"""
        # 这里模拟强化学习的行为，实际上是根据时间动态调整音乐生成参数
        
        # 创建分段音频
        segments = 4
        segment_duration = duration / segments
        # 修改分段生成部分，添加instruments处理
        instruments_full = np.zeros(int(duration * self.sample_rate))
        
        # 在循环内生成每个片段的instruments
        for i in range(segments):
        
            # 结果音频
            melody_full = np.zeros(int(duration * self.sample_rate))
            harmony_full = np.zeros(int(duration * self.sample_rate))
            percussion_full = np.zeros(int(duration * self.sample_rate))
            ambient_full = np.zeros(int(duration * self.sample_rate))
            
            # 随着时间动态调整生成参数
            for i in range(segments):
                # 模拟不同时段的参数调整
                segment_params = self.params.copy()
                
                # 模拟根据"环境反馈"调整参数
                if i == 1:
                    # 第二段增加一些复杂度
                    segment_params["rhythm_complexity"] = min(1.0, segment_params["rhythm_complexity"] * 1.2)
                    segment_params["harmony_density"] = min(1.0, segment_params["harmony_density"] * 1.1)
                elif i == 2:
                    # 第三段增加强度
                    segment_params["volume"] = min(1.0, segment_params["volume"] * 1.15)
                    segment_params["tempo"] = min(200, segment_params["tempo"] * 1.05)
                elif i == 3:
                    # 最后一段回归平静
                    segment_params["rhythm_complexity"] = segment_params["rhythm_complexity"] * 0.9
                    segment_params["reverb"] = min(0.9, segment_params["reverb"] * 1.2)
                
                # 根据调整后的参数创建临时生成器
                melody_gen = MelodyGenerator(segment_params)
                harmony_gen = HarmonyGenerator(segment_params)
                percussion_gen = PercussionGenerator(segment_params)
                ambient_gen = AmbientGenerator(segment_params)
                
                # 生成片段
                melody_segment = melody_gen.generate(segment_duration)
                harmony_segment = harmony_gen.generate(segment_duration)
                percussion_segment = percussion_gen.generate(segment_duration)
                ambient_segment = ambient_gen.generate(segment_duration)
                
                # 添加到完整音频
                start = int(i * segment_duration * self.sample_rate)
                end = int((i+1) * segment_duration * self.sample_rate)
                end = min(end, len(melody_full))
                
                melody_full[start:end] = melody_segment[:end-start]
                harmony_full[start:end] = harmony_segment[:end-start]
                percussion_full[start:end] = percussion_segment[:end-start]
                ambient_full[start:end] = ambient_segment[:end-start]
                instruments_segment = self._generate_instruments_track(segment_duration)
                
                # 添加到完整音频
                instruments_full[start:end] = instruments_segment[:end-start]
                
                # 处理过渡，添加乐器轨道的淡变
                if i < segments - 1:
                    # 过渡区长度 (0.2秒)
                    crossfade = int(0.2 * self.sample_rate)
                    crossfade_start = end - crossfade
                    
                    # 生成下一段开头，用于交叉淡入淡出
                    if i < segments - 1:
                        # 预生成下一段的开头
                        next_melody = melody_gen.generate(0.3)[:crossfade]
                        next_harmony = harmony_gen.generate(0.3)[:crossfade]
                        next_percussion = percussion_gen.generate(0.3)[:crossfade]
                        next_ambient = ambient_gen.generate(0.3)[:crossfade]
                        next_instruments = self._generate_instruments_track(0.3)[:crossfade]
                        
                        # 创建淡出/淡入包络
                        fadeout = np.linspace(1, 0, crossfade)
                        fadein = np.linspace(0, 1, crossfade)
                        
                        # 应用交叉淡变
                        melody_full[crossfade_start:end] = melody_full[crossfade_start:end] * fadeout + next_melody * fadein
                        harmony_full[crossfade_start:end] = harmony_full[crossfade_start:end] * fadeout + next_harmony * fadein
                        percussion_full[crossfade_start:end] = percussion_full[crossfade_start:end] * fadeout + next_percussion * fadein
                        ambient_full[crossfade_start:end] = ambient_full[crossfade_start:end] * fadeout + next_ambient * fadein
                        instruments_full[crossfade_start:end] = instruments_full[crossfade_start:end] * fadeout + next_instruments * fadein
        
        # 混合各轨道
        result = (
            melody_full * 0.7 + 
            harmony_full * 0.5 + 
            percussion_full * 0.6 + 
            ambient_full * 0.4 +
            instruments_full * 0.5
        )
        
        # 标准化
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return {
            "combined": result,
            "melody": melody_full,
            "harmony": harmony_full, 
            "percussion": percussion_full,
            "ambient": ambient_full,
            "instruments": instruments_full
        }
    
    def generate(self, duration=10.0, collab_mode="parallel"):
        """生成音乐，根据指定的协作模式"""
        if collab_mode == "pipeline":
            return self.generate_pipeline(duration)
        elif collab_mode == "hybrid":
            return self.generate_hybrid(duration)
        elif collab_mode == "rl":
            return self.generate_rl(duration)
        else:  # 默认使用并行模式
            return self.generate_parallel(duration)
    
    def save_audio(self, audio_data, file_path, sample_rate=44100):
        """保存音频到文件"""
        # 转换为 int16
        audio_normalized = audio_data / np.max(np.abs(audio_data))
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存文件
        wavfile.write(file_path, sample_rate, audio_int16)

# 从文本中提取情感参数
def extract_emotion_from_text(text):
    """从文本中提取情感参数（简单实现，实际项目中可能需要NLP模型）"""
    text = text.lower()
    
    # 定义情感关键词
    emotion_keywords = {
        "joy": ["快乐", "喜悦", "幸福", "欢乐", "开心", "高兴", "欣喜", "愉快", "兴奋", "雀跃"],
        "sadness": ["悲伤", "忧愁", "悲痛", "伤心", "难过", "哀伤", "郁闷", "沮丧", "消沉", "痛苦"],
        "fear": ["恐惧", "惊恐", "害怕", "惊吓", "畏惧", "恐慌", "不安", "担忧", "怯懦", "胆怯"],
        "anger": ["愤怒", "生气", "恼火", "发怒", "气愤", "暴怒", "激怒", "恼怒", "怒火", "愤慨"],
        "surprise": ["惊讶", "震惊", "意外", "诧异", "愕然", "惊奇", "惊愕", "惊诧", "惊讶", "吃惊"],
        "disgust": ["厌恶", "恶心", "反感", "嫌弃", "憎恨", "憎恶", "讨厌", "鄙视", "蔑视", "唾弃"],
        "anticipation": ["期待", "期盼", "盼望", "希望", "憧憬", "企盼", "预期", "展望", "等待", "渴望"],
        "trust": ["信任", "相信", "信赖", "依靠", "信奉", "信心", "依赖", "相信", "信念", "信服"]
    }
    
    # 分析文本中的情感关键词
    emotion_scores = {emotion: 0 for emotion in emotion_keywords}
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 1
    
    # 寻找主要情感
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    if dominant_emotion[1] == 0:
        # 如果没有找到明确情感，根据文本设置默认情感
        if any(word in text for word in ["战斗", "紧张", "危险", "战争", "敌人"]):
            dominant_emotion = ("fear", 1)
        elif any(word in text for word in ["平静", "放松", "安宁", "平和", "宁静"]):
            dominant_emotion = ("trust", 1)
        else:
            dominant_emotion = ("anticipation", 1)  # 默认为期待
    
    # 估算效价和唤醒度
    valence = 0.5  # 默认中性
    arousal = 0.5  # 默认中性
    
    # 根据情感类型设置效价和唤醒度
    if dominant_emotion[0] in ["joy", "anticipation", "trust"]:
        valence = 0.7 + random.random() * 0.3  # 高效价
    elif dominant_emotion[0] in ["sadness", "disgust"]:
        valence = 0.2 + random.random() * 0.3  # 低效价
    
    if dominant_emotion[0] in ["fear", "anger", "surprise"]:
        arousal = 0.7 + random.random() * 0.3  # 高唤醒
    elif dominant_emotion[0] in ["sadness", "trust"]:
        arousal = 0.2 + random.random() * 0.3  # 低唤醒
    
    # 基于文本内容进一步调整
    intensity_words = ["非常", "极其", "格外", "特别", "十分", "异常", "格外", "尤为", "极为", "尤其"]
    intensity = 0.6  # 默认中等强度
    
    for word in intensity_words:
        if word in text:
            intensity += 0.1
            intensity = min(intensity, 1.0)
    
    if any(word in text for word in ["战斗", "战争", "危机", "冲突"]):
        arousal = min(1.0, arousal + 0.2)
    
    if any(word in text for word in ["安静", "平和", "宁静", "祥和"]):
        arousal = max(0.0, arousal - 0.2)
    
    return {
        "emotion": dominant_emotion[0],
        "valence": valence,
        "arousal": arousal,
        "intensity": intensity
    }

def batch_generate_music_files(output_dir, duration=10.0):
    """批量生成音乐文件"""
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建音乐资源目录结构
    dirs = ["melody", "harmony", "percussion", "ambient", "instruments"]
    for dir_name in dirs:
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    # 情感类型
    emotions = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "anticipation", "trust"]
    
    # 速度类别
    speeds = ["fast", "medium", "slow"]
    
    # 情感速度映射
    emotion_speed_map = {
        "joy": ["fast", "medium"],
        "sadness": ["slow", "medium"],
        "fear": ["fast", "medium"],
        "anger": ["fast"],
        "surprise": ["fast", "medium"],
        "disgust": ["slow", "medium"],
        "anticipation": ["medium"],
        "trust": ["medium", "slow"]
    }
    
    # 协作模式
    collab_modes = ["parallel", "pipeline", "hybrid", "rl"]
    
    # 情感映射到效价/唤醒度
    emotion_va_map = {
        "joy": (0.8, 0.7),
        "sadness": (0.2, 0.3),
        "fear": (0.3, 0.8),
        "anger": (0.2, 0.9),
        "surprise": (0.6, 0.8),
        "disgust": (0.2, 0.5),
        "anticipation": (0.7, 0.6),
        "trust": (0.7, 0.4)
    }
    
    # 生成队列
    generation_queue = []
    
    # 为每种情感/速度组合生成音乐
    for emotion in emotions:
        valence, arousal = emotion_va_map[emotion]
        
        # 生成正面负面情绪映射
        if emotion in ["joy", "anticipation", "trust"]:
            mood = "positive"
        elif emotion in ["sadness", "disgust", "fear"]:
            mood = "negative"
        else:
            mood = "neutral"
        
        # 为每种速度生成旋律
        for speed in emotion_speed_map.get(emotion, ["medium"]):
            # 调整参数使其符合速度
            speed_arousal = arousal
            if speed == "fast":
                speed_arousal = min(1.0, arousal + 0.2)
            elif speed == "slow":
                speed_arousal = max(0.1, arousal - 0.2)
            
            # 创建任务
            task = {
                "emotion": emotion,
                "mood": mood,
                "speed": speed,
                "valence": valence,
                "arousal": speed_arousal,
                "intensity": 0.8,
                "duration": duration,
                "collab_mode": random.choice(collab_modes)  # 随机选择协作模式
            }
            generation_queue.append(task)
    
    # 创建几个预设的乐器音色示例
    instruments = ["piano", "strings", "synth", "flute", "guitar"]
    moods = ["positive", "negative"]
    
    for instrument in instruments:
        for mood in moods:
            # 为每个乐器/情绪组合创建一个生成任务
            if mood == "positive":
                emotion = "joy"
                valence, arousal = emotion_va_map[emotion]
            else:
                emotion = "sadness"
                valence, arousal = emotion_va_map[emotion]
                
            task = {
                "emotion": emotion,
                "mood": mood,
                "instrument": instrument,
                "valence": valence,
                "arousal": arousal,
                "intensity": 0.8,
                "duration": duration,
                "collab_mode": "emotion_focused"
            }
            generation_queue.append(task)
    
    # 创建默认文件
    default_tasks = [
        {
            "type": "default_melody",
            "emotion": "anticipation",
            "mood": "neutral",
            "valence": 0.5,
            "arousal": 0.5,
            "intensity": 0.7,
            "duration": duration,
            "collab_mode": "parallel"
        },
        {
            "type": "default_harmony",
            "emotion": "trust",
            "mood": "neutral",
            "valence": 0.5,
            "arousal": 0.4,
            "intensity": 0.6,
            "duration": duration,
            "collab_mode": "parallel"
        },
        {
            "type": "default_percussion",
            "emotion": "anticipation",
            "mood": "neutral",
            "valence": 0.5,
            "arousal": 0.5,
            "intensity": 0.7,
            "duration": duration,
            "collab_mode": "parallel"
        }
    ]
    generation_queue.extend(default_tasks)
    
    # 打印生成任务信息
    print(f"准备生成 {len(generation_queue)} 个音乐文件...")
    
    # 执行生成任务
    for i, task in enumerate(generation_queue):
        print(f"[{i+1}/{len(generation_queue)}] 生成: {task.get('type', task.get('emotion'))} - {task.get('speed', '')} {task.get('instrument', '')}")
        
        # 创建音乐生成器
        if "type" not in task:  # 正常情感音乐
            generator = MusicGenerator(
                emotion=task["emotion"],
                valence=task["valence"],
                arousal=task["arousal"],
                intensity=task["intensity"]
            )
            
            # 生成音乐
            audio_tracks = generator.generate(
                duration=task["duration"],
                collab_mode=task["collab_mode"]
            )
            
            # 处理乐器音色文件
            if "instrument" in task:
                # 为乐器保存音频
                filename = f"{task['instrument']}_{task['mood']}.wav"
                output_path = os.path.join(output_dir, "instruments", filename)
                generator.save_audio(audio_tracks["combined"], output_path)
                continue
            
            # 保存旋律文件
            if "speed" in task:  # 只有旋律文件有速度标记
                for track_idx in range(1, 3):  # 每种组合生成两个文件
                    filename = f"{task['mood']}_melody_{task['speed']}_{track_idx}.wav"
                    output_path = os.path.join(output_dir, "melody", filename)
                    generator.save_audio(audio_tracks["melody"], output_path)
            
            # 保存和声文件
            for track_idx in range(1, 3):  # 每种情绪生成两个文件
                filename = f"{task['mood']}_harmony_{track_idx}.wav"
                output_path = os.path.join(output_dir, "harmony", filename)
                generator.save_audio(audio_tracks["harmony"], output_path)
            
            # 保存打击乐文件（如果有速度）
            if "speed" in task:
                for track_idx in range(1, 3):  # 每种速度生成两个文件
                    filename = f"percussion_{task['speed']}_{track_idx}.wav"
                    output_path = os.path.join(output_dir, "percussion", filename)
                    generator.save_audio(audio_tracks["percussion"], output_path)
            
            # 保存环境音文件
            filename = f"ambient_{task['mood']}.wav"
            output_path = os.path.join(output_dir, "ambient", filename)
            generator.save_audio(audio_tracks["ambient"], output_path)
        
        else:  # 默认文件
            generator = MusicGenerator(
                emotion=task["emotion"],
                valence=task["valence"],
                arousal=task["arousal"],
                intensity=task["intensity"]
            )
            
            # 生成音乐
            audio_tracks = generator.generate(
                duration=task["duration"],
                collab_mode=task["collab_mode"]
            )
            
            # 根据类型保存默认文件
            if task["type"] == "default_melody":
                output_path = os.path.join(output_dir, "melody", "default_melody.wav")
                generator.save_audio(audio_tracks["melody"], output_path)
            elif task["type"] == "default_harmony":
                output_path = os.path.join(output_dir, "harmony", "default_harmony.wav")
                generator.save_audio(audio_tracks["harmony"], output_path)
            elif task["type"] == "default_percussion":
                output_path = os.path.join(output_dir, "percussion", "default_percussion.wav")
                generator.save_audio(audio_tracks["percussion"], output_path)
    
    print(f"音乐文件生成完成，共 {len(generation_queue)} 个文件已保存到 {output_dir}")
    return True

# 命令行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成情感音乐文件")
    parser.add_argument("--output", default="data/music", help="输出目录")
    parser.add_argument("--duration", type=float, default=10.0, help="音频时长(秒)")
    parser.add_argument("--emotion", default=None, help="指定情感(可选)")
    parser.add_argument("--speed", default=None, help="指定速度(可选)")
    
    args = parser.parse_args()
    
    if args.emotion:
        # 生成单个指定情感的音乐
        valence, arousal = 0.5, 0.5  # 默认值
        
        # 尝试设置合适的效价和唤醒度
        emotion_va_map = {
            "joy": (0.8, 0.7),
            "sadness": (0.2, 0.3),
            "fear": (0.3, 0.8),
            "anger": (0.2, 0.9),
            "surprise": (0.6, 0.8),
            "disgust": (0.2, 0.5),
            "anticipation": (0.7, 0.6),
            "trust": (0.7, 0.4)
        }
        
        if args.emotion in emotion_va_map:
            valence, arousal = emotion_va_map[args.emotion]
        
        # 调整速度对应的唤醒度
        if args.speed == "fast":
            arousal = min(1.0, arousal + 0.2)
        elif args.speed == "slow":
            arousal = max(0.1, arousal - 0.2)
        
        # 设置情绪类型
        if args.emotion in ["joy", "anticipation", "trust"]:
            mood = "positive"
        elif args.emotion in ["sadness", "disgust", "fear"]:
            mood = "negative"
        else:
            mood = "neutral"
        
        generator = MusicGenerator(
            emotion=args.emotion,
            valence=valence,
            arousal=arousal,
            intensity=0.8
        )
        
        audio_tracks = generator.generate(duration=args.duration)
        
        # 保存音频文件
        os.makedirs(args.output, exist_ok=True)
        
        # 如果指定了速度，则使用速度信息命名
        speed_info = f"_{args.speed}" if args.speed else ""
        output_path = os.path.join(args.output, f"{args.emotion}{speed_info}_music.wav")
        generator.save_audio(audio_tracks["combined"], output_path)
        
        print(f"已生成情感音乐: {output_path}")
    else:
        # 批量生成所有音乐文件
        batch_generate_music_files(args.output, args.duration)