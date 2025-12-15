#####################################
# 情感分析函数
#####################################
#####################################
# 高级情感分析引擎
#####################################
import re
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import deque
import time
import hashlib
import functools
import logging
from enum import Enum, auto
import enum

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emotion_analyzer")

# 情感分析模型懒加载管理
class ModelLoader:
    _instance = None
    _lock = threading.Lock()
    _models = {}
    _initializing = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_model(self, model_name: str) -> Any:
        """智能加载模型，支持懒加载和资源管理"""
        if model_name in self._models:
            return self._models[model_name]
            
        with self._lock:
            # 双重检查锁定模式
            if model_name in self._models:
                return self._models[model_name]
                
            # 基于模型类型动态加载
            if model_name == 'vader':
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                try:
                    import nltk
                    nltk.download('vader_lexicon', quiet=True)
                except:
                    pass
                self._models[model_name] = SentimentIntensityAnalyzer()
                
            elif model_name == 'transformers':
                try:
                    from transformers import pipeline
                    # 使用DistilBERT以平衡性能和准确性
                    self._models[model_name] = pipeline('sentiment-analysis', 
                                                        model='distilbert-base-uncased-finetuned-sst-2-english',
                                                        return_all_scores=True)
                except ImportError:
                    logger.warning("transformers库未安装，无法加载高级情感模型")
                    self._models[model_name] = None
                    
            elif model_name == 'emotion_bert':
                try:
                    from transformers import pipeline
                    # 加载情感专用模型
                    self._models[model_name] = pipeline('text-classification', 
                                                        model='j-hartmann/emotion-english-distilroberta-base',
                                                        return_all_scores=True)
                except ImportError:
                    logger.warning("transformers库未安装，无法加载情感分类模型")
                    self._models[model_name] = None
            
            elif model_name == 'nrclex':
                try:
                    from nrclex import NRCLex
                    self._models[model_name] = NRCLex
                except ImportError:
                    logger.warning("nrclex库未安装，无法加载NRC情感词典")
                    self._models[model_name] = None
                
        return self._models.get(model_name)

# 结果缓存装饰器
def cache_sentiment_results(max_size=1000, ttl=3600):
    cache = {}
    timestamps = {}
    cache_lock = threading.Lock()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 区分实例方法和普通函数调用
            if args and hasattr(args[0], '__class__') and not isinstance(args[0], str):
                # 这是一个实例方法调用，第一个参数是self
                self = args[0]
                text = args[1] if len(args) > 1 else kwargs.get('text', '')
            else:
                # 普通函数调用
                text = args[0] if args else kwargs.get('text', '')
            
            # 创建指纹以识别相似文本
            if isinstance(text, str):
                if len(text) > 1000:  # 长文本使用哈希
                    text_fingerprint = hashlib.md5(text.encode('utf-8')).hexdigest()
                else:
                    # 短文本直接使用
                    text_fingerprint = text
            else:
                # 不是字符串类型，无法缓存
                return func(*args, **kwargs)
                
            cache_key = (text_fingerprint, str(args[1:] if args and hasattr(args[0], '__class__') else args), str(kwargs))
            
            with cache_lock:
                # 清理过期缓存
                current_time = time.time()
                expired_keys = [k for k, t in timestamps.items() if current_time - t > ttl]
                for k in expired_keys:
                    if k in cache:
                        del cache[k]
                    if k in timestamps:
                        del timestamps[k]
                
                # 查找缓存
                if cache_key in cache:
                    return cache[cache_key]
            
            # 缓存未命中，执行函数
            result = func(*args, **kwargs)
            
            with cache_lock:
                # 如果缓存太大，移除最旧的项
                if len(cache) >= max_size:
                    oldest_key = min(timestamps.items(), key=lambda x: x[1])[0]
                    del cache[oldest_key]
                    del timestamps[oldest_key]
                
                # 添加到缓存
                cache[cache_key] = result
                timestamps[cache_key] = current_time
                
            return result
        return wrapper
    return decorator

def get_emotion_analyzer(config=None):
    """获取情感分析器实例"""
    return EmotionAnalyzer(config)

class EmotionDimension(Enum):
    """情感维度枚举"""
    VALENCE = auto()     # 情感效价
    AROUSAL = auto()     # 情感唤醒
    DOMINANCE = auto()   # 情感支配
    SURPRISE = auto()    # 惊奇
    CERTAINTY = auto()   # 确定性
    ATTENTION = auto()   # 注意力
    INTENSITY = auto()   # 强度

@dataclass
class EmotionVector:
    """多维情感向量，支持多种情感表示"""
    valence: float = 0.5       # 情感效价 (-1.0 到 1.0)
    arousal: float = 0.5       # 情感唤醒度 (0.0 到 1.0)
    dominance: float = 0.5     # 支配度
    surprise: float = 0.0      # 惊奇度
    certainty: float = 0.5     # 确定性
    attention: float = 0.5     # 注意力
    intensity: float = 0.5     # 强度
    confidence: float = 1.0    # 分析置信度
    
    raw_scores: Dict[str, float] = field(default_factory=dict)
    emotion_categories: Dict[str, float] = field(default_factory=dict)
    
    def to_valence_arousal(self) -> Tuple[float, float]:
        """转换为简单的valence-arousal二维表示"""
        # 归一化valence到0-1范围
        norm_valence = (self.valence + 1) / 2
        return (norm_valence, self.arousal)
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组表示"""
        return np.array([
            self.valence, self.arousal, self.dominance,
            self.surprise, self.certainty, self.attention, self.intensity
        ])
    
    def blend(self, other: 'EmotionVector', weight: float = 0.5) -> 'EmotionVector':
        """与另一个情感向量混合"""
        result = EmotionVector()
        for field_name, field_value in vars(self).items():
            if isinstance(field_value, (int, float)) and field_name != 'confidence':
                other_value = getattr(other, field_name)
                setattr(result, field_name, field_value * (1 - weight) + other_value * weight)
        
        # 混合情感类别
        all_categories = set(list(self.emotion_categories.keys()) + list(other.emotion_categories.keys()))
        for category in all_categories:
            self_value = self.emotion_categories.get(category, 0.0)
            other_value = other.emotion_categories.get(category, 0.0)
            result.emotion_categories[category] = self_value * (1 - weight) + other_value * weight
            
        # 置信度取最低值
        result.confidence = min(self.confidence, other.confidence)
        return result

@dataclass
class StoryEmotionAnalysisResult:
    """故事情感分析结果，包含全局和局部情感信息"""
    global_emotion: EmotionVector  # 整体情感
    segment_emotions: List[EmotionVector] = field(default_factory=list)  # 分段情感
    emotional_arc: List[Tuple[float, float]] = field(default_factory=list)  # 情感弧线(valence, arousal)
    key_moments: List[Dict[str, Any]] = field(default_factory=list)  # 关键情感时刻
    emotional_variance: float = 0.0  # 情感变化程度
    emotional_complexity: float = 0.0  # 情感复杂度
    language_intensity: float = 0.0  # 语言强度

class EmotionAnalyzer:
    """高级情感分析引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._model_loader = ModelLoader.get_instance()
        self._sentence_cache = {}
        self._paragraph_history = deque(maxlen=5)  # 保留最近分析的段落历史
        
        # 配置默认值
        self.use_transformers = self.config.get('use_transformers', True)
        self.use_lexicon = self.config.get('use_lexicon', True)
        self.detect_arcs = self.config.get('detect_arcs', True)
        self.segment_min_length = self.config.get('segment_min_length', 100)
        
    def get_vader_analyzer(self):
        """获取VADER分析器"""
        return self._model_loader.get_model('vader')
    
    def get_transformer_analyzer(self):
        """获取Transformer分析器"""
        return self._model_loader.get_model('transformers')
    
    def get_emotion_classifier(self):
        """获取情感分类器"""
        return self._model_loader.get_model('emotion_bert')
    
    def get_nrc_lexicon(self):
        """获取NRC情感词典"""
        return self._model_loader.get_model('nrclex')
        
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """将分数归一化到0-1范围"""
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))
    
    def _segment_text(self, text: str) -> List[str]:
        """智能分段文本"""
        # 基于段落分隔
        paragraphs = re.split(r'\n\s*\n', text)
        
        segments = []
        current_segment = ""
        current_length = 0
        
        for para in paragraphs:
            para_trimmed = para.strip()
            if not para_trimmed:
                continue
                
            para_length = len(para_trimmed)
            
            # 如果段落本身很长，可能需要进一步分割
            if para_length > self.segment_min_length * 2:
                # 分割长段落
                sentences = re.split(r'[.!?]+', para_trimmed)
                sentence_group = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(sentence_group) + len(sentence) < self.segment_min_length:
                        if sentence_group:
                            sentence_group += ". " + sentence
                        else:
                            sentence_group = sentence
                    else:
                        if sentence_group:
                            segments.append(sentence_group + ".")
                        sentence_group = sentence
                
                if sentence_group:
                    segments.append(sentence_group + ".")
            else:
                # 合并短段落
                if current_length + para_length < self.segment_min_length:
                    if current_segment:
                        current_segment += "\n\n" + para_trimmed
                    else:
                        current_segment = para_trimmed
                    current_length += para_length
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = para_trimmed
                    current_length = para_length
        
        # 添加最后一个段落
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    @cache_sentiment_results(max_size=500)
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """使用VADER分析文本情感"""
        analyzer = self.get_vader_analyzer()
        if not analyzer:
            return {'pos': 0, 'neg': 0, 'neu': 0.5, 'compound': 0}
            
        try:
            return analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"VADER分析出错: {str(e)}")
            return {'pos': 0, 'neg': 0, 'neu': 0.5, 'compound': 0}
    
    def _analyze_with_transformers(self, text: str) -> Dict[str, float]:
        """使用Transformers模型分析文本"""
        analyzer = self.get_transformer_analyzer()
        if not analyzer:
            return {'positive': 0.5, 'negative': 0.5}
            
        try:
            # 截断过长文本
            if len(text) > 512:
                text = text[:512]
                
            results = analyzer(text)
            scores = {item['label']: item['score'] for item in results[0]}
            return scores
        except Exception as e:
            logger.error(f"Transformers分析出错: {str(e)}")
            return {'positive': 0.5, 'negative': 0.5}
    
    def _analyze_with_emotion_model(self, text: str) -> Dict[str, float]:
        """使用专门的情感分类模型"""
        classifier = self.get_emotion_classifier()
        if not classifier:
            return {}
            
        try:
            # 截断过长文本
            if len(text) > 512:
                text = text[:512]
                
            results = classifier(text)
            emotions = {item['label']: item['score'] for item in results[0]}
            return emotions
        except Exception as e:
            logger.error(f"情感分类模型分析出错: {str(e)}")
            return {}
    
    def _analyze_with_lexicon(self, text: str) -> Dict[str, float]:
        """使用NRC情感词典分析"""
        NRCLex = self.get_nrc_lexicon()
        if not NRCLex:
            return {}
            
        try:
            text_object = NRCLex(text)
            affect_dict = text_object.affect_dict
            
            # 计算各情感得分
            emotion_counts = {}
            total_words = 0
            
            for word, emotions in affect_dict.items():
                word_count = text.lower().count(word.lower())
                total_words += word_count
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + word_count
            
            # 标准化
            if total_words > 0:
                return {emotion: count / total_words for emotion, count in emotion_counts.items()}
            else:
                return {}
        except Exception as e:
            logger.error(f"词典情感分析出错: {str(e)}")
            return {}
    
    def _calculate_linguistic_features(self, text: str) -> Dict[str, float]:
        """计算语言学特征"""
        features = {}
        
        # 句子长度变异度
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
        
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            features['avg_sentence_length'] = avg_length
            
            if len(sentence_lengths) > 1:
                variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                features['sentence_length_variance'] = variance
            else:
                features['sentence_length_variance'] = 0
                
        # 感叹号、问号数量
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # 大写字母比例
        if len(text) > 0:
            uppercase_chars = sum(1 for c in text if c.isupper())
            features['uppercase_ratio'] = uppercase_chars / len(text)
        else:
            features['uppercase_ratio'] = 0
            
        # 情感词汇密度通过上面的词典分析已经得到
            
        return features
    
    def _extract_emotional_arc(self, segment_emotions: List[EmotionVector]) -> List[Dict[str, Any]]:
        """提取情感弧线中的关键点"""
        if len(segment_emotions) < 2:
            return []
            
        key_points = []
        va_sequence = [e.to_valence_arousal() for e in segment_emotions]
        
        # 找到局部极值点
        for i in range(1, len(va_sequence) - 1):
            prev_v, prev_a = va_sequence[i-1]
            curr_v, curr_a = va_sequence[i]
            next_v, next_a = va_sequence[i+1]
            
            # 效价局部峰值
            if (curr_v > prev_v and curr_v > next_v) or (curr_v < prev_v and curr_v < next_v):
                key_points.append({
                    'index': i,
                    'type': 'valence_peak' if curr_v > prev_v else 'valence_valley',
                    'score': curr_v,
                    'magnitude': abs(curr_v - (prev_v + next_v) / 2)
                })
                
            # 唤醒度局部峰值
            if (curr_a > prev_a and curr_a > next_a) or (curr_a < prev_a and curr_a < next_a):
                key_points.append({
                    'index': i,
                    'type': 'arousal_peak' if curr_a > prev_a else 'arousal_valley',
                    'score': curr_a,
                    'magnitude': abs(curr_a - (prev_a + next_a) / 2)
                })
        
        # 找到全局极值点
        if va_sequence:
            max_v_idx = max(range(len(va_sequence)), key=lambda i: va_sequence[i][0])
            min_v_idx = min(range(len(va_sequence)), key=lambda i: va_sequence[i][0])
            max_a_idx = max(range(len(va_sequence)), key=lambda i: va_sequence[i][1])
            min_a_idx = min(range(len(va_sequence)), key=lambda i: va_sequence[i][1])
            
            key_points.extend([
                {'index': max_v_idx, 'type': 'global_max_valence', 'score': va_sequence[max_v_idx][0]},
                {'index': min_v_idx, 'type': 'global_min_valence', 'score': va_sequence[min_v_idx][0]},
                {'index': max_a_idx, 'type': 'global_max_arousal', 'score': va_sequence[max_a_idx][1]},
                {'index': min_a_idx, 'type': 'global_min_arousal', 'score': va_sequence[min_a_idx][1]}
            ])
            
        # 按重要性排序
        return sorted(key_points, key=lambda x: x.get('magnitude', 0), reverse=True)
        
    def _compute_emotional_complexity(self, emotion_vectors: List[EmotionVector]) -> float:
        """计算情感复杂度"""
        if not emotion_vectors:
            return 0.0
            
        # 分析情感种类数量和分布
        all_categories = set()
        for ev in emotion_vectors:
            all_categories.update(ev.emotion_categories.keys())
            
        diversity = len(all_categories) / 8  # 假设最多识别8种基本情感
        
        # 计算情感变化率
        if len(emotion_vectors) >= 2:
            changes = []
            for i in range(1, len(emotion_vectors)):
                prev = emotion_vectors[i-1].to_array()
                curr = emotion_vectors[i].to_array()
                change = np.linalg.norm(curr - prev)
                changes.append(change)
                
            avg_change = sum(changes) / len(changes)
            max_change = max(changes)
            
            # 复杂度综合考虑情感多样性和变化率
            complexity = (diversity * 0.5) + (avg_change * 0.3) + (max_change * 0.2)
            return min(1.0, complexity * 2)  # 归一化到0-1
            
        return diversity * 0.8  # 仅有一个样本，主要基于多样性
    
    def analyze_text_emotion(self, text: str, detailed: bool = False) -> EmotionVector:
        """
        分析文本情感，返回多维情感向量
        
        参数:
            text: 输入文本
            detailed: 是否返回详细分析
            
        返回:
            EmotionVector 情感向量
        """
        emotion_vector = EmotionVector()
        confidence = 1.0
        
        # 空文本快速返回
        if not text or not text.strip():
            emotion_vector.confidence = 0.0
            return emotion_vector
        
        # 1. 使用VADER进行基础情感分析
        vader_scores = self._analyze_with_vader(text)
        
        valence = vader_scores['compound']  # 范围: [-1, 1]
        arousal = 1 - vader_scores['neu']   # 中性值越低，唤醒度越高
        
        # 2. 使用Transformer模型提升精度（如果配置允许）
        if self.use_transformers:
            try:
                transformer_scores = self._analyze_with_transformers(text)
                
                # 整合Transformer结果（给予更高权重）
                if transformer_scores:
                    t_valence = transformer_scores.get('positive', 0.5) - transformer_scores.get('negative', 0.5)  # 转为[-1, 1]
                    valence = valence * 0.4 + t_valence * 2 * 0.6  # 给transformer更高权重
                    
                # 情感分类增强
                emotion_scores = self._analyze_with_emotion_model(text)
                if emotion_scores:
                    # 使用特定情感强度计算唤醒度
                    excitement = emotion_scores.get('joy', 0) + emotion_scores.get('surprise', 0)
                    calm = emotion_scores.get('sadness', 0) + emotion_scores.get('fear', 0) * 0.5
                    t_arousal = excitement / (excitement + calm) if (excitement + calm) > 0 else 0.5
                    
                    # 整合结果
                    arousal = arousal * 0.3 + t_arousal * 0.7
                    
                    # 更新情感类别
                    emotion_vector.emotion_categories = emotion_scores
            except Exception as e:
                confidence *= 0.8  # 降低置信度
                logger.warning(f"Transformer情感分析失败: {str(e)}")
        
        # 3. 使用词典增强（如果配置允许）
        if self.use_lexicon:
            try:
                lexicon_scores = self._analyze_with_lexicon(text)
                
                if lexicon_scores:
                    # 计算支配度
                    dominance = lexicon_scores.get('trust', 0.5) - lexicon_scores.get('fear', 0)
                    dominance = max(0, min(1, dominance + 0.5))  # 调整到[0, 1]范围
                    emotion_vector.dominance = dominance
                    
                    # 计算惊奇度
                    surprise = lexicon_scores.get('surprise', 0)
                    emotion_vector.surprise = surprise
                    
                    # 更细致的情感类别补充
                    for emotion, score in lexicon_scores.items():
                        emotion_vector.emotion_categories[emotion] = max(
                            emotion_vector.emotion_categories.get(emotion, 0),
                            score
                        )
            except Exception as e:
                confidence *= 0.9  # 轻微降低置信度
                logger.warning(f"词典情感分析失败: {str(e)}")
        
        # 4. 语言学特征分析
        ling_features = self._calculate_linguistic_features(text)
        
        # 使用感叹号等调整唤醒度和强度
        intensity_factor = min(1.0, (
            ling_features.get('exclamation_count', 0) * 0.1 +
            ling_features.get('question_count', 0) * 0.05 +
            ling_features.get('uppercase_ratio', 0) * 2
        ))
        
        emotion_vector.intensity = min(1.0, arousal * 0.7 + intensity_factor * 0.3)
        
        # 考虑句子长度对确定性的影响
        if 'sentence_length_variance' in ling_features:
            variance = ling_features['sentence_length_variance']
            # 变异度高表示表达不确定
            certainty = 0.8 - min(0.6, variance / 100)
            emotion_vector.certainty = max(0, min(1, certainty))
        
        # 设置基本情感维度
        emotion_vector.valence = max(-1.0, min(1.0, valence))
        emotion_vector.arousal = max(0.0, min(1.0, arousal))
        emotion_vector.confidence = confidence
        
        # 保存原始分数
        emotion_vector.raw_scores = vader_scores
        
        return emotion_vector
    
    def analyze_story_emotions(self, story_text: str, segment: bool = True) -> StoryEmotionAnalysisResult:
        """
        分析故事情感，支持全文分析和分段分析
        
        参数:
            story_text: 故事文本
            segment: 是否分段分析
            
        返回:
            StoryEmotionAnalysisResult 全文和分段情感分析结果
        """
        # 全文分析
        global_emotion = self.analyze_text_emotion(story_text, detailed=True)
        
        # 初始化结果
        result = StoryEmotionAnalysisResult(global_emotion=global_emotion)
        
        # 文本分段
        if segment and len(story_text) > self.segment_min_length:
            segments = self._segment_text(story_text)
            
            # 分析每个段落
            for seg_text in segments:
                segment_emotion = self.analyze_text_emotion(seg_text, detailed=True)
                result.segment_emotions.append(segment_emotion)

            # 计算情感弧线
            result.emotional_arc = [em.to_valence_arousal() for em in result.segment_emotions]

            # 提取关键情感时刻
            result.key_moments = self._extract_emotional_arc(result.segment_emotions)

            # 计算情感变化程度 (方差示例)
            if len(result.segment_emotions) > 1:
                all_valences = [em.valence for em in result.segment_emotions]
                result.emotional_variance = float(np.var(all_valences))

            # 计算情感复杂度
            result.emotional_complexity = self._compute_emotional_complexity(result.segment_emotions)

            # 计算语言强度 (取段落强度平均值)
            if result.segment_emotions:
                intensities = [em.intensity for em in result.segment_emotions]
                result.language_intensity = float(np.mean(intensities))

        return result

#####################################
# 情感分析函数 - 高级智能情感分析
#####################################
def analyze_story_emotions(story_text):
    """
    使用高级情感分析引擎分析故事情节的情感。
    支持多维情感模型，提供更准确的分析结果。
    
    Args:
        story_text: 用户输入的故事情节文本
    
    Returns:
        valence和arousal值的元组
    """
    try:
        # 尝试使用高级情感分析
        analyzer = get_emotion_analyzer()
        
        # 获取完整分析结果
        result = analyzer.analyze_story_emotions(story_text)
        
        # 提取全局情感向量
        emotion_vector = result.global_emotion
        
        # 转换为简单的valence-arousal表示
        valence, arousal = emotion_vector.to_valence_arousal()
        
        # 增强功能：利用情感复杂度调整arousal
        # 复杂的情感通常更具激活性
        if result.emotional_complexity > 0:
            arousal = (arousal * 0.7) + (result.emotional_complexity * 0.3)
            arousal = min(1.0, arousal)  # 确保不超过1.0
        
        # 记录详细情感分析
        logger.info(f"故事情感分析结果: valence={valence:.2f}, arousal={arousal:.2f}")
        logger.info(f"情感复杂度: {result.emotional_complexity:.2f}, 情感变化: {result.emotional_variance:.2f}")
        
        # 返回标准化的valence和arousal
        return valence, arousal
        
    except Exception as e:
        logger.error(f"高级情感分析失败，使用基础分析: {e}")
        
        # 回退到基础VADER分析
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # 尝试下载词典资源（如果还没有的话）
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass
            
            # 初始化VADER分析器
            vader_analyzer = SentimentIntensityAnalyzer()
            
            # 分析文本获取基础情感分数
            vader_scores = vader_analyzer.polarity_scores(story_text)
            
            # 从compound分数获取valence(效价)
            valence = (vader_scores['compound'] + 1) / 2  # 转换为0-1范围
            
            # 从neu分数推断arousal(唤醒度)
            arousal = 1 - vader_scores['neu']  # 中性越低，唤醒度越高
            
            # 记录基础分析结果
            logger.info(f"基础VADER情感分析结果: valence={valence:.2f}, arousal={arousal:.2f}")
            
            return valence, arousal
            
        except Exception as e:
            # 如果连VADER分析也失败，使用默认值
            logger.error(f"基础VADER分析也失败: {e}, 使用默认值")
            return 0.5, 0.5