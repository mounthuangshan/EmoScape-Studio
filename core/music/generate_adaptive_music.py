
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import math
import time
import joblib # 用于保存和加载模型
import music21
from music21 import stream, note, chord, instrument, tempo, key, meter, scale, tempo, metadata, pitch as m21pitch
import magenta.music as mm
from magenta.models.music_vae import configs as vae_configs
from magenta.models.music_vae.trained_model import TrainedModel as VaeTrainedModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pygame  # 用于播放MIDI和WAV文件
#pygame.mixer.init()
import threading  # 用于在后台播放音乐
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, AutoModelForCausalLM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
from deepface import DeepFace
import cv2
import json
import asyncio
import websockets
from bigvgan_v2_44khz_128band_512x import bigvgan
from pathlib import Path
import copy
import subprocess
from datetime import datetime
import librosa

#项目文件
from core.emotional.plutchik_get_emotion_model import *
from core.emotional.train_emotion_mapping_model import *
from core.emotional.simulate_emotion_stream import *
from core.emotional.analyze_emotion import *
from core.music.choose_functional_chord_progression import *
from core.music.get_advanced_culture_settings import *
from core.music.music_gan import *
from core.music.music_vae import *
from core.music.music_transformer import *
from core.music.melody_processor import *
from core.music.note_num_to_note_name import *
from core.music.audio_effects import *
from utils.play_audio import *
from utils.create_lyrics import *
from utils.visualization import *
from utils.evaluation_quality import *
from utils.db_and_recommendation import *
from core.generate_structured_music import *

#####################################
# 自适应音乐生成系统
#####################################
class AdaptiveModel:
    """自适应模型，根据用户反馈历史预测最佳音乐生成参数"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameter_weights = {
            'temperature': {'base': 1.0, 'weight': 0.0},
            'note_density': {'base': 8, 'weight': 0.0},
            'harmony_complexity': {'base': 1, 'weight': 0.0},
            'rhythm_intensity': {'base': 0.5, 'weight': 0.0},
            'melodic_range': {'base': 0.6, 'weight': 0.0}
        }
        self.valence_influence = 0.5  # valence对各参数的影响权重
        self.arousal_influence = 0.5  # arousal对各参数的影响权重
        
    def train(self, feedback_history):
        """
        使用历史反馈训练模型
        
        :param feedback_history: 历史反馈数据列表，每项包含情感值、生成参数和用户评分
        :return: 训练后的模型
        """
        if not feedback_history:
            print("没有历史反馈数据，使用默认参数")
            return self
            
        print(f"使用{len(feedback_history)}条反馈历史训练自适应模型")
        
        # 按评分排序，优先考虑高评分样本
        sorted_feedback = sorted(feedback_history, key=lambda x: x['rating'], reverse=True)
        
        # 提取高评分样本中的情感-参数关系
        high_rated = [item for item in sorted_feedback if item['rating'] > 0.7]
        
        if not high_rated:
            print("没有高评分样本，使用所有样本")
            high_rated = sorted_feedback
            
        # 计算参数与情感值的相关性
        for param_name in self.parameter_weights:
            valence_correlation = 0
            arousal_correlation = 0
            
            for item in high_rated:
                if param_name in item['parameters']:
                    param_value = item['parameters'][param_name]
                    param_norm = self._normalize_param(param_name, param_value)
                    
                    # 简化相关性计算
                    valence_correlation += (item['emotion']['valence'] - 0.5) * (param_norm - 0.5) * item['rating']
                    arousal_correlation += (item['emotion']['arousal'] - 0.5) * (param_norm - 0.5) * item['rating']
            
            # 更新参数权重
            self.valence_influence = max(0.0, min(1.0, 0.5 + valence_correlation))
            self.arousal_influence = max(0.0, min(1.0, 0.5 + arousal_correlation))
            
            # 更新参数基线
            if high_rated:
                # 计算高评分样本中的参数平均值
                param_avg = sum(item['parameters'].get(param_name, self.parameter_weights[param_name]['base']) 
                              for item in high_rated) / len(high_rated)
                
                # 调整基线值，逐渐靠近高评分样本的平均值
                self.parameter_weights[param_name]['base'] = (
                    0.8 * self.parameter_weights[param_name]['base'] + 
                    0.2 * param_avg
                )
                
        print("自适应模型训练完成")
        return self
                
    def predict(self, emotion_stream):
        """
        预测适合当前情感流的音乐参数
        
        :param emotion_stream: 情感值流，包含PlutchikEmotion对象列表
        :return: 针对每个情感状态预测的参数字典列表
        """
        predictions = []
        
        for emotion in emotion_stream:
            valence, arousal = emotion.to_valence_arousal()
            
            # 预测各参数
            params = {}
            for param_name, settings in self.parameter_weights.items():
                base_value = settings['base']
                
                # 根据情感状态调整参数
                valence_adj = (valence - 0.5) * self.valence_influence
                arousal_adj = (arousal - 0.5) * self.arousal_influence
                
                # 不同参数对情感的响应不同
                if param_name == 'temperature':
                    # 温度受arousal影响更大
                    adjusted = base_value * (1 + arousal_adj * 0.4 - valence_adj * 0.1)
                    params[param_name] = max(0.6, min(1.2, adjusted))
                    
                elif param_name == 'note_density':
                    # 音符密度受arousal影响更大
                    adjusted = base_value * (1 + arousal_adj * 0.5)
                    params[param_name] = max(4, min(16, int(adjusted)))
                    
                elif param_name == 'harmony_complexity':
                    # 和声复杂度受valence和arousal共同影响
                    complexity_base = 1
                    if arousal > 0.7:
                        complexity_base = 3
                    elif arousal > 0.4:
                        complexity_base = 2
                    # valence高时略微增加复杂度
                    if valence > 0.7 and complexity_base < 3:
                        complexity_base += 1
                    params[param_name] = complexity_base
                    
                elif param_name == 'rhythm_intensity':
                    # 节奏强度主要受arousal影响
                    adjusted = base_value + arousal_adj * 0.6
                    params[param_name] = max(0.1, min(1.0, adjusted))
                    
                elif param_name == 'melodic_range':
                    # 旋律范围受valence影响较大
                    adjusted = base_value + valence_adj * 0.3 + arousal_adj * 0.2
                    params[param_name] = max(0.2, min(1.0, adjusted))
            
            # 记录当前情感值
            params['emotion'] = {'valence': valence, 'arousal': arousal}
            predictions.append(params)
            
        return predictions
    
    def update(self, feedback):
        """
        根据新反馈实时更新模型
        
        :param feedback: 新的反馈数据
        :return: 更新后的模型
        """
        # 如果评分较高，则更新参数权重
        if feedback['rating'] > 0.7:
            for param_name, param_value in feedback['parameters'].items():
                if param_name in self.parameter_weights:
                    # 更新参数基线，向成功参数靠近
                    old_base = self.parameter_weights[param_name]['base']
                    new_base = old_base + self.learning_rate * (param_value - old_base)
                    self.parameter_weights[param_name]['base'] = new_base
                    
        return self
    
    def _normalize_param(self, param_name, value):
        """
        将参数值归一化到0-1范围
        
        :param param_name: 参数名称
        :param value: 参数值
        :return: 归一化的参数值(0-1)
        """
        if param_name == 'temperature':
            return (value - 0.6) / 0.6  # 0.6-1.2 -> 0-1
        elif param_name == 'note_density':
            return (value - 4) / 12  # 4-16 -> 0-1
        elif param_name == 'harmony_complexity':
            return (value - 1) / 2  # 1-3 -> 0-1
        elif param_name in ['rhythm_intensity', 'melodic_range']:
            return value  # 已在0-1范围内
        return 0.5  # 默认中间值

def train_adaptive_model(feedback_history):
    """
    基于历史生成和用户反馈创建适应性模型
    
    :param feedback_history: 用户反馈历史记录
    :return: 训练好的适应性模型
    """
    model = AdaptiveModel()
    return model.train(feedback_history)

def generate_with_adaptive_params(emotion_stream, predicted_params, vae_model=None, 
                                harmony_complexity=1, culture="Western", feedback_callback=None):
    """
    使用自适应预测的参数生成音乐
    
    :param emotion_stream: 情感流，包含PlutchikEmotion对象列表
    :param predicted_params: 预测的参数列表，与emotion_stream长度相同
    :param vae_model: MusicVAE模型
    :param harmony_complexity: 基础和声复杂度
    :param culture: 文化背景
    :param feedback_callback: 反馈回调函数，用于实时调整
    :return: 生成的结构化音乐
    """
    # 确保PlutchikEmotion类已扩展
    extend_plutchik_emotion()
    
    if not emotion_stream or not predicted_params:
        print("错误: 缺少情感流或预测参数")
        return None
    
    # 为每个部分选择适当的情感和参数
    intro_idx = 0  # 使用开始的情感状态
    verse_idx = len(emotion_stream) // 4  # 第一个四分之一
    chorus_idx = len(emotion_stream) // 2  # 中间情感状态
    bridge_idx = int(len(emotion_stream) * 0.75)  # 第三个四分之一
    outro_idx = -1  # 使用结束的情感状态
    
    # 获取各部分的参数
    intro_params = predicted_params[intro_idx]
    verse_params = predicted_params[verse_idx]
    chorus_params = predicted_params[chorus_idx]
    bridge_params = predicted_params[bridge_idx]
    outro_params = predicted_params[outro_idx]
    
    # 设置各部分长度
    structure_params = {
        "intro_length": max(2, min(8, intro_params.get('length', 4))),
        "verse_length": max(4, min(16, verse_params.get('length', 8))),
        "chorus_length": max(4, min(16, chorus_params.get('length', 8))),
        "bridge_length": max(2, min(8, bridge_params.get('length', 4))),
        "outro_length": max(2, min(8, outro_params.get('length', 4)))
    }
    
    print(f"自适应生成结构化音乐: 文化={culture}")
    print(f"结构参数: 前奏={structure_params['intro_length']}小节, "
        f"主歌={structure_params['verse_length']}小节, "
        f"副歌={structure_params['chorus_length']}小节, "
        f"桥段={structure_params['bridge_length']}小节, "
        f"尾奏={structure_params['outro_length']}小节")
    
    # 创建各段落, 使用各情感点的预测参数
    intro = generate_intro(
        emotion_stream[intro_idx], 
        structure_params["intro_length"], 
        vae_model, 
        harmony_complexity=intro_params.get('harmony_complexity', harmony_complexity),
        temperature=intro_params.get('temperature', 0.8),
        culture=culture
    )
    
    verse = generate_verse(
        emotion_stream[verse_idx], 
        structure_params["verse_length"], 
        vae_model, 
        harmony_complexity=verse_params.get('harmony_complexity', harmony_complexity),
        temperature=verse_params.get('temperature', 0.9),
        culture=culture
    )
    
    # 副歌使用增强的情感
    chorus = generate_chorus(
        emotion_stream[chorus_idx].intensify(), 
        structure_params["chorus_length"], 
        vae_model, 
        harmony_complexity=chorus_params.get('harmony_complexity', harmony_complexity + 1),
        temperature=chorus_params.get('temperature', 0.7),
        culture=culture
    )
    
    # 桥段使用过渡情感
    bridge = generate_bridge(
        emotion_stream[bridge_idx].transition(), 
        structure_params["bridge_length"], 
        vae_model, 
        harmony_complexity=bridge_params.get('harmony_complexity', harmony_complexity + 1),
        temperature=bridge_params.get('temperature', 1.0),
        culture=culture
    )
    
    outro = generate_outro(
        emotion_stream[outro_idx], 
        structure_params["outro_length"], 
        vae_model, 
        harmony_complexity=outro_params.get('harmony_complexity', harmony_complexity),
        temperature=outro_params.get('temperature', 0.7),
        culture=culture
    )
    
    # 实时反馈调整（如果提供了回调）
    if feedback_callback:
        feedback = feedback_callback(intro)
        if feedback and 'parameters' in feedback:
            intro = adjust_section(intro, feedback['parameters'])
        
        feedback = feedback_callback(verse)
        if feedback and 'parameters' in feedback:
            verse = adjust_section(verse, feedback['parameters'])
            
        feedback = feedback_callback(chorus)
        if feedback and 'parameters' in feedback:
            chorus = adjust_section(chorus, feedback['parameters'])
            
        feedback = feedback_callback(bridge)
        if feedback and 'parameters' in feedback:
            bridge = adjust_section(bridge, feedback['parameters'])
            
        feedback = feedback_callback(outro)
        if feedback and 'parameters' in feedback:
            outro = adjust_section(outro, feedback['parameters'])
    
    # 按照结构组装完整曲目
    full_piece = assemble_music_sections([
        intro, verse, chorus, verse, chorus, bridge, chorus, outro
    ], with_transitions=True)
    
    # 设置曲目元信息
    if hasattr(full_piece, 'metadata') and full_piece.metadata:
        avg_valence = sum(e.to_valence_arousal()[0] for e in emotion_stream) / len(emotion_stream)
        avg_arousal = sum(e.to_valence_arousal()[1] for e in emotion_stream) / len(emotion_stream)
        
        emotion_desc = "欢快" if avg_valence > 0.7 else "忧伤" if avg_valence < 0.3 else "平静"
        intensity_desc = "激昂" if avg_arousal > 0.7 else "舒缓" if avg_arousal < 0.3 else "中等"
        
        full_piece.metadata.title = f"自适应{emotion_desc}而{intensity_desc}的音乐"
        full_piece.metadata.composer = "AI自适应作曲家"
    
    return full_piece

def adjust_section(section, params):
    """
    根据反馈参数调整音乐段落
    
    :param section: 音乐段落(music21.stream.Score)
    :param params: 调整参数
    :return: 调整后的段落
    """
    # 调整音量
    if 'volume' in params:
        volume_scale = params['volume']
        for part in section.parts:
            for note_obj in part.flat.notes:
                if hasattr(note_obj, 'volume'):
                    note_obj.volume.velocity = int(note_obj.volume.velocity * volume_scale)
    
    # 调整速度
    if 'tempo' in params:
        tempo_scale = params['tempo']
        for tempo_mark in section.flat.getElementsByClass('MetronomeMark'):
            new_tempo = tempo_mark.number * tempo_scale
            tempo_mark.number = new_tempo
    
    # 不修改原始数据，返回复制品
    return section

def adaptive_music_generation(emotion_stream, feedback_history, vae_model=None, transformer_model=None, 
                            culture="Western", harmony_complexity=1):
    """
    基于历史生成和用户反馈创建适应性音乐生成系统
    
    :param emotion_stream: 情感流，包含PlutchikEmotion对象列表
    :param feedback_history: 用户反馈历史
    :param vae_model: MusicVAE模型
    :param transformer_model: Transformer模型
    :param culture: 文化背景
    :param harmony_complexity: 基础和声复杂度
    :return: 生成的自适应音乐
    """
    generating = True
    new_feedback = None
    feedback_queue = queue.Queue()
    
    # 训练自适应模型
    adaptive_model = train_adaptive_model(feedback_history)
    
    # 预测用户偏好的音乐参数
    predicted_params = adaptive_model.predict(emotion_stream)
    
    # 创建反馈回调函数
    def feedback_callback(section):
        if not feedback_queue.empty():
            new_feedback = feedback_queue.get()
            # 更新自适应模型
            adaptive_model.update(new_feedback)
            return new_feedback
        return None
    
    # 生成自适应音乐
    try:
        music = generate_with_adaptive_params(
            emotion_stream, 
            predicted_params, 
            vae_model=vae_model, 
            harmony_complexity=harmony_complexity, 
            culture=culture,
            feedback_callback=feedback_callback
        )
    except Exception as e:
        print(f"生成自适应音乐时出错: {e}")
        # 创建一个简单的空白音乐对象作为后备
        from music21 import stream, tempo, meter, key
        music = stream.Score()
        music.append(tempo.MetronomeMark(number=120))
        music.append(meter.TimeSignature('4/4'))
        music.append(key.Key('C'))
    
    generating = False
    return music

def generate_adaptive_music_handler(gui_instance):
    """处理生成自适应结构化音乐的请求"""
    try:
        valence = gui_instance.valence_slider.get()
        arousal = gui_instance.arousal_slider.get()
        cultures = [culture for culture, var in gui_instance.culture_vars.items() if var.get()]
        
        if not cultures:
            messagebox.showerror("错误", "请选择至少一个文化背景。")
            return
            
        # 获取用户选择的乐器
        additional_instruments = [instr for instr, var in gui_instance.instrument_vars.items() if var.get()]
        
        # 和声复杂度
        harmony_complexity = gui_instance.harmony_complexity.get()
        
        # 创建情感对象
        emotion = PlutchikEmotion(
            joy=valence + gui_instance.user_feedback.get("adjust_valence", 0),
            trust=0.5,
            fear=1 - (arousal + gui_instance.user_feedback.get("adjust_arousal", 0)),
            surprise=0.3,
            sadness=0.2,
            disgust=0.1,
            anger=0.1,
            anticipation=0.4
        )
        
        # 显示进度
        gui_instance.progress['value'] = 10
        gui_instance.root.update_idletasks()
        
        # 生成多个情感点
        emotion_stream = [emotion]
        for i in range(5):
            # 稍微变化情感值
            new_emotion = PlutchikEmotion(
                joy=max(0, min(1, emotion.joy + random.uniform(-0.2, 0.2))),
                trust=max(0, min(1, emotion.trust + random.uniform(-0.1, 0.1))),
                fear=max(0, min(1, emotion.fear + random.uniform(-0.2, 0.2))),
                surprise=max(0, min(1, emotion.surprise + random.uniform(-0.1, 0.1))),
                sadness=max(0, min(1, emotion.sadness + random.uniform(-0.1, 0.1))),
                disgust=max(0, min(1, emotion.disgust + random.uniform(-0.1, 0.1))),
                anger=max(0, min(1, emotion.anger + random.uniform(-0.1, 0.1))),
                anticipation=max(0, min(1, emotion.anticipation + random.uniform(-0.1, 0.1)))
            )
            emotion_stream.append(new_emotion)
        
        # 收集用户反馈历史
        feedback_history = load_user_feedback_history()
        
        # 生成自适应音乐
        culture = cultures[0]  # 使用第一个选择的文化
        print(f"生成自适应音乐: 文化={culture}, 和声复杂度={harmony_complexity}")
        
        # 启动线程生成音乐
        threading.Thread(
            target=_generate_adaptive_music_thread,
            args=(gui_instance, emotion_stream, feedback_history, harmony_complexity, culture, additional_instruments)
        ).start()
        
    except Exception as e:
        messagebox.showerror("错误", f"生成自适应音乐时出错: {e}")
        gui_instance.progress['value'] = 0
        gui_instance.root.update_idletasks()

def _generate_adaptive_music_thread(gui_instance, emotion_stream, feedback_history, harmony_complexity, culture, additional_instruments):
    """在后台线程中生成自适应音乐"""
    try:
        gui_instance.progress['value'] = 20
        gui_instance.root.update_idletasks()
        
        # 生成自适应音乐
        full_piece = adaptive_music_generation(
            emotion_stream,
            feedback_history,
            vae_model=gui_instance.vae_model,
            transformer_model=gui_instance.transformer_model,
            harmony_complexity=harmony_complexity,
            culture=culture
        )
        
        gui_instance.progress['value'] = 60
        gui_instance.root.update_idletasks()
        
        # 保存为MIDI文件
        filename = "adaptive_music.mid"
        full_piece.write("midi", fp=filename)
        print(f"已保存自适应音乐: {filename}")
        
        # 转换为WAV并添加音频效果
        wav_path = "./output/adaptive_music.wav"
        processed_wav_path = "./output/adaptive_music_processed.wav"
        
        # 确保目录存在
        os.makedirs("output", exist_ok=True)
        
        # 转换MIDI到WAV
        convert_midi_to_wav(filename, wav_path, soundfont_path=gui_instance.soundfont_path)
        
        gui_instance.progress['value'] = 80
        gui_instance.root.update_idletasks()
        
        # 获取平均情感值
        avg_valence = sum(e.to_valence_arousal()[0] for e in emotion_stream) / len(emotion_stream)
        avg_arousal = sum(e.to_valence_arousal()[1] for e in emotion_stream) / len(emotion_stream)
        
        # 应用音频效果
        process_audio(wav_path, processed_wav_path, avg_valence, avg_arousal)
        
        # 清理临时文件
        try:
            os.remove(filename)
            os.remove(wav_path)
        except:
            pass
        
        # 生成歌词
        lyrics = generate_lyrics(gui_instance.lyrics_tokenizer, gui_instance.lyrics_model, emotion_stream[0])
        
        # 在主线程中更新UI
        gui_instance.root.after(0, lambda: _update_ui_after_adaptive_generation(gui_instance, processed_wav_path, lyrics))
        
    except Exception as e:
        gui_instance.root.after(0, lambda: messagebox.showerror("错误", f"生成自适应音乐时出错: {e}"))
        gui_instance.root.after(0, lambda: setattr(gui_instance.progress, 'value', 0))

def _update_ui_after_adaptive_generation(gui_instance, output_file, lyrics):
    """在自适应音乐生成完成后更新UI"""
    gui_instance.progress['value'] = 100
    gui_instance.root.update_idletasks()
    
    # 显示歌词
    gui_instance.lyrics_text.delete(1.0, tk.END)
    gui_instance.lyrics_text.insert(tk.END, lyrics)
    
    # 显示提示
    messagebox.showinfo("完成", f"已生成自适应音乐: {output_file}")
    
    # 启用播放按钮
    gui_instance.play_button.config(state='normal')
    gui_instance.currently_playing = output_file
    
    # 请求用户反馈
    show_adaptive_feedback_dialog(gui_instance, output_file)

def show_adaptive_feedback_dialog(gui_instance, music_file):
    """显示自适应音乐反馈对话框"""
    feedback_window = tk.Toplevel(gui_instance.root)
    feedback_window.title("自适应音乐反馈")
    
    # 评分滑块
    ttk.Label(feedback_window, text="请为生成的自适应音乐评分:").pack(pady=10)
    
    # 整体评分
    ttk.Label(feedback_window, text="整体质量:").pack(pady=5)
    overall_var = tk.DoubleVar(value=0.5)
    overall_scale = ttk.Scale(feedback_window, from_=0, to=1, orient='horizontal', length=200, variable=overall_var)
    overall_scale.pack(pady=5)
    
    # 情感匹配度
    ttk.Label(feedback_window, text="情感匹配度:").pack(pady=5)
    emotion_var = tk.DoubleVar(value=0.5)
    emotion_scale = ttk.Scale(feedback_window, from_=0, to=1, orient='horizontal', length=200, variable=emotion_var)
    emotion_scale.pack(pady=5)
    
    # 结构评价
    ttk.Label(feedback_window, text="结构设计:").pack(pady=5)
    structure_var = tk.DoubleVar(value=0.5)
    structure_scale = ttk.Scale(feedback_window, from_=0, to=1, orient='horizontal', length=200, variable=structure_var)
    structure_scale.pack(pady=5)
    
    # 特定部分评价
    section_frame = ttk.LabelFrame(feedback_window, text="各部分评价")
    section_frame.pack(pady=10, fill=tk.X, padx=20)
    
    section_vars = {}
    for section in ["前奏", "主歌", "副歌", "桥段", "尾奏"]:
        ttk.Label(section_frame, text=f"{section}:").pack(pady=2)
        section_vars[section] = tk.DoubleVar(value=0.5)
        ttk.Scale(section_frame, from_=0, to=1, orient='horizontal', length=180, variable=section_vars[section]).pack(pady=2)
    
    # 调整建议
    ttk.Label(feedback_window, text="调整建议:").pack(pady=5)
    suggestion_text = tk.Text(feedback_window, height=4, width=40)
    suggestion_text.pack(pady=5)
    
    # 保存反馈
    def save_feedback():
        # 获取当前的valence和arousal值
        valence = gui_instance.valence_slider.get()
        arousal = gui_instance.arousal_slider.get()
        
        # 创建反馈数据
        feedback_data = {
            'music_file': music_file,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'emotion': {
                'valence': valence,
                'arousal': arousal
            },
            'rating': overall_var.get(),
            'emotion_match': emotion_var.get(),
            'structure_rating': structure_var.get(),
            'section_ratings': {
                section: var.get() for section, var in section_vars.items()
            },
            'suggestion': suggestion_text.get('1.0', tk.END).strip(),
            'parameters': {
                'temperature': 0.8 + arousal * 0.4,
                'note_density': 8 + int(arousal * 4),
                'harmony_complexity': 1 if valence < 0.5 else 2 if arousal < 0.7 else 3,
                'rhythm_intensity': arousal,
                'melodic_range': 0.5 + valence * 0.2
            }
        }
        
        # 使用全局函数而不是类方法
        save_user_feedback(feedback_data)  # 删除self.前缀
        
        feedback_window.destroy()
        messagebox.showinfo("反馈", "感谢您的反馈！这将帮助系统提升音乐生成质量。")
    
    # 保存按钮
    ttk.Button(feedback_window, text="提交反馈", command=save_feedback).pack(pady=10)


# 修复全局函数
def save_user_feedback(feedback_data):
    """保存用户反馈到文件"""
    feedback_file = "data/databases/user_feedback_adaptive.json"
    
    # 读取现有反馈
    existing_feedback = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                existing_feedback = json.load(f)
        except:
            pass
    
    # 添加新反馈
    existing_feedback.append(feedback_data)
    
    # 保存更新后的反馈
    with open(feedback_file, 'w') as f:
        json.dump(existing_feedback, f, indent=2)
    
    print(f"已保存用户反馈到 {feedback_file}")

def load_user_feedback_history():
    """加载用户反馈历史"""
    feedback_file = "data/databases/user_feedback_adaptive.json"
    
    if not os.path.exists(feedback_file):
        return []
    
    try:
        with open(feedback_file, 'r') as f:
            feedback_history = json.load(f)
        print(f"已加载{len(feedback_history)}条用户反馈历史")
        return feedback_history
    except Exception as e:
        print(f"加载用户反馈历史时出错: {e}")
        return []