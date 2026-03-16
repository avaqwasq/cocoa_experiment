#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoCoA Uncertainty Quantification Experiment
согласно статье arXiv:2502.04964

CoCoA: Multiple samples + confidence × consistency
CoCoA Light: MLP на эмбеддингах среднего слоя + greedy decoding
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Конфигурация эксперимента согласно статье"""
    
    # Модель
    # Варианты (чем меньше — тем быстрее):
    #   "Qwen/Qwen2.5-0.5B-Instruct"  — самый быстрый, ~3× быстрее 1.5B
    #   "Qwen/Qwen2.5-1.5B-Instruct"  — баланс скорость/качество
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    # Параметры для CoCoA Light (эмбеддинги)
    # Qwen2.5-0.5B: 24 слоя, hidden_size=896, берём слой 12 (середина)
    # Qwen2.5-1.5B: 28 слоёв, hidden_size=1536, берём слой 14 (середина)
    EMBEDDING_LAYER = 12
    EMBEDDING_DIM = 896   # 896 для 0.5B, 1536 для 1.5B

    # Параметры генерации
    MAX_NEW_TOKENS = 64   # 128 избыточно для QA; 64 вдвое быстрее
    TEMPERATURE = 0.7
    TOP_P = 0.9
    NUM_SAMPLES = 5       # 10 по статье, но 5 достаточно для проверки метода
    
    # Датасеты
    DATASETS = {
        "TriviaQA": {"name": "trivia_qa", "subset": "rc.wikipedia.nocontext", 
                     "size": 300, "task_type": "qa"},
        "CoQA": {"name": "coqa", "subset": None, "size": 300, "task_type": "qa"},
        "XSUM": {"name": "xsum", "subset": None, "size": 300, "task_type": "summarization"},
    }
    
    # Датасет для обучения CoCoA Light (unlabeled held-out set)
    LIGHT_TRAIN_SIZE = 200  # В статье: 3000-10000; 200 достаточно для проверки
    
    # Функция сходства
    SIMILARITY_MODEL = "sentence-transformers/LaBSE"
    
    # Устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4-bit квантование
    USE_4BIT = False
    
    # Режимы
    RUN_COCOA = True       # Запустить полный CoCoA
    RUN_COCOA_LIGHT = True # Запустить CoCoA Light с обучением MLP

# ============================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================================
def load_llm_model(model_name: str, use_4bit: bool = True):
    """Загружает языковую модель с квантованием"""
    
    print(f"Загрузка модели: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if use_4bit and torch.cuda.is_available():
        # Создаём конфигурацию квантования
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,  # ← Передаём через quantization_config
            trust_remote_code=True,
            output_hidden_states=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
        )
    
    model.eval()
    print(f"Модель загружена на {next(model.parameters()).device}")
    return model, tokenizer

def load_similarity_model(model_name: str):
    """Загружает модель для вычисления семантического сходства"""
    print(f"Загрузка модели сходства: {model_name}")
    similarity_model = SentenceTransformer(model_name)
    print(f"Модель сходства загружена")
    return similarity_model

# ============================================================================
# ЗАГРУЗКА ДАТАСЕТОВ
# ============================================================================

def load_mini_dataset(dataset_name: str, dataset_config: Dict) -> List[Dict]:
    """Загружает мини-версию датасета"""
    
    task_type = dataset_config["task_type"]
    size = dataset_config["size"]
    
    print(f"Загрузка датасета: {dataset_name} ({task_type}, {size} примеров)")
    
    data = []
    
    if task_type == "qa":
        if dataset_name == "TriviaQA":
            dataset = load_dataset("trivia_qa", "rc.wikipedia.nocontext", split="validation")
            for i, item in enumerate(dataset):
                if i >= size:
                    break
                data.append({
                    "input": item["question"],
                    "reference": item["answer"]["value"],
                    "task_type": "qa"
                })
        
        elif dataset_name == "CoQA":
            dataset = load_dataset("coqa", split="validation")
            for i, item in enumerate(dataset):
                if i >= size:
                    break
                question = item["questions"][-1]
                answer = item["answers"]["input_text"][-1]
                data.append({
                    "input": question,
                    "reference": answer,
                    "task_type": "qa"
                })
    
    elif task_type == "summarization":
        dataset = load_dataset("xsum", split="test")
        for i, item in enumerate(dataset):
            if i >= size:
                break
            data.append({
                "input": item["document"],
                "reference": item["summary"],
                "task_type": "summarization"
            })
    
    elif task_type == "translation":
        if dataset_name == "IWSLT_EN_DE":
            dataset = load_dataset("iwslt2017", "en-de", split="test")
            for i, item in enumerate(dataset):
                if i >= size:
                    break
                data.append({
                    "input": item["translation"]["en"],
                    "reference": item["translation"]["de"],
                    "task_type": "translation",
                    "src_lang": "en",
                    "tgt_lang": "de"
                })
    
    print(f"Загружено {len(data)} примеров")
    return data

# ============================================================================
# ФОРМИРОВАНИЕ ПРОМПТОВ
# ============================================================================

def create_prompt(item: Dict) -> str:
    """Создаёт промпт в зависимости от типа задачи"""
    
    task_type = item["task_type"]
    input_text = item["input"]
    
    if task_type == "qa":
        prompt = f"Question: {input_text}\n\nAnswer:"
    
    elif task_type == "summarization":
        prompt = f"Summarize the following text:\n\n{input_text}\n\nSummary:"
    
    elif task_type == "translation":
        src_lang = item.get("src_lang", "en")
        tgt_lang = item.get("tgt_lang", "de")
        lang_names = {"en": "English", "de": "German", "fr": "French", 
                      "es": "Spanish", "ru": "Russian", "zh": "Chinese"}
        prompt = f"Translate from {lang_names.get(src_lang, src_lang)} to {lang_names.get(tgt_lang, tgt_lang)}:\n\n{input_text}\n\nTranslation:"
    
    else:
        prompt = input_text
    
    return prompt

# ============================================================================
# ГЕНЕРАЦИЯ ОТВЕТОВ С ЭМБЕДДИНГАМИ
# ============================================================================

def generate_responses_with_embeddings(model, tokenizer, prompt: str,
                                        num_samples: int, config: Config,
                                        return_embeddings: bool = False):
    """
    Генерирует multiple samples и возвращает:
    - (текст, log_probability) для CoCoA
    - эмбеддинги среднего слоя для CoCoA Light

    Все samples генерируются за ОДИН батчевый вызов model.generate().
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(config.DEVICE)
    input_length = inputs["input_ids"].shape[1]

    # Дублируем вход для батчевой генерации всех N samples за один вызов
    batch_inputs = {k: v.repeat(num_samples, 1) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=return_embeddings,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    embeddings_list = []

    for i in range(num_samples):
        # Декодируем ответ для i-го sample
        generated_ids = outputs.sequences[i][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Вычисляем log probability для i-го sample
        if hasattr(outputs, 'scores') and outputs.scores:
            log_probs = []
            for t, score in enumerate(outputs.scores):
                token_id = outputs.sequences[i][input_length + t]
                log_prob = torch.log_softmax(score[i], dim=0)[token_id].item()
                log_probs.append(log_prob)
            avg_log_prob = np.mean(log_probs) if log_probs else 0.0
        else:
            avg_log_prob = 0.0

        responses.append((text, avg_log_prob))

        # Извлекаем эмбеддинги среднего слоя (для CoCoA Light)
        if return_embeddings:
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states and len(outputs.hidden_states) > config.EMBEDDING_LAYER:
                layer_hidden = outputs.hidden_states[config.EMBEDDING_LAYER][-1]
                embedding = layer_hidden[i, -1, :].cpu().float().numpy()
                embeddings_list.append(embedding)
            else:
                embeddings_list.append(None)  # Keep in sync with responses

    return responses, embeddings_list if return_embeddings else None

def generate_greedy_with_embedding(model, tokenizer, prompt: str, config: Config):
    """
    Генерирует ОДИН greedy ответ с эмбеддингами (для CoCoA Light inference)
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                       max_length=512).to(config.DEVICE)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Декодируем ответ
    generated_ids = outputs.sequences[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Вычисляем log probability
    if hasattr(outputs, 'scores') and outputs.scores:
        log_probs = []
        for i, score in enumerate(outputs.scores):
            token_id = outputs.sequences[0][input_length + i]
            log_prob = torch.log_softmax(score[0], dim=0)[token_id].item()
            log_probs.append(log_prob)
        avg_log_prob = np.mean(log_probs) if log_probs else 0.0
    else:
        avg_log_prob = 0.0
    
    # Извлекаем эмбеддинги среднего слоя
    embedding = None
    if outputs.hidden_states and len(outputs.hidden_states) > config.EMBEDDING_LAYER:
        layer_hidden = outputs.hidden_states[config.EMBEDDING_LAYER][-1]
        embedding = layer_hidden[0, -1, :].cpu().float().numpy()
    
    return (text, avg_log_prob), embedding

# ============================================================================
# COCOA LIGHT: MLP МОДЕЛЬ
# ============================================================================

class CoCoALightMLP:
    """
    MLP для предсказания consistency uncertainty из эмбеддингов
    Согласно Appendix I статьи
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 2048):
        # Используем MLPRegressor из sklearn (проще чем PyTorch для этой задачи)
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='relu',
            solver='adam',
            alpha=0.1,  # Weight decay
            batch_size=4,
            learning_rate_init=1e-5,
            max_iter=20,  # 20 epochs
            early_stopping=False,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, embeddings: np.ndarray, targets: np.ndarray):
        """
        Обучает MLP на эмбеддингах и целевых consistency uncertainty
        
        Args:
            embeddings: [n_samples, embedding_dim]
            targets: [n_samples] - true consistency uncertainty
        """
        print(f"Обучение CoCoA Light MLP...")
        print(f"   Размер training set: {len(embeddings)} примеров")
        print(f"   Размерность эмбеддингов: {embeddings.shape[1]}")
        
        # Нормализуем эмбеддинги
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Обучаем модель
        self.model.fit(embeddings_scaled, targets)
        self.is_trained = True
        
        # Оценка качества
        train_score = self.model.score(embeddings_scaled, targets)
        print(f"MLP обучена (R² = {train_score:.4f})")
    
    def predict(self, embedding: np.ndarray) -> float:
        """Предсказывает consistency uncertainty из одного эмбеддинга"""
        
        if not self.is_trained:
            raise ValueError("MLP не обучена! Вызовите train() сначала.")
        
        embedding_scaled = self.scaler.transform([embedding])
        prediction = self.model.predict(embedding_scaled)[0]
        
        # Нормализуем к [0, 1]
        return float(np.clip(prediction, 0.0, 1.0))

# ============================================================================
# МЕТРИКИ НЕОПРЕДЕЛЁННОСТИ
# ============================================================================

def compute_sequence_probability(responses: List[Tuple[str, float]]) -> float:
    """U_SP: Sequence Probability (information-based)"""
    
    if not responses:
        return 1.0
    
    best_log_prob = max(r[1] for r in responses)
    probability = np.exp(best_log_prob)
    uncertainty = 1.0 - min(probability, 1.0)
    return uncertainty

def compute_perplexity(responses: List[Tuple[str, float]]) -> float:
    """U_PPL: Perplexity-based uncertainty"""
    
    if not responses:
        return 1.0
    
    best_log_prob = max(r[1] for r in responses)
    perplexity = np.exp(-best_log_prob)
    uncertainty = min(perplexity / 100.0, 1.0)
    return uncertainty

def compute_semantic_similarity(model, text1: str, text2: str) -> float:
    """Вычисляет семантическое сходство между двумя текстами"""
    
    if not text1.strip() or not text2.strip():
        return 0.0
    
    try:
        embeddings = model.encode([text1, text2], convert_to_numpy=True, 
                                  show_progress_bar=False)
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        similarity = (similarity + 1) / 2
        return float(similarity)
    except:
        return 0.0

def compute_consistency_uncertainty(similarity_model, responses: List[Tuple[str, float]]) -> float:
    """
    U_cons: Consistency-based uncertainty (Formula 8 в статье)

    Û_cons(y*|x) = (1/M) × Σ(1 - s(y*, yⁱ))

    Все тексты кодируются за один батчевый вызов encode().
    """

    if len(responses) < 2:
        return 1.0

    texts = [r[0] for r in responses]
    best_text = max(responses, key=lambda x: x[1])[0]  # y*

    other_texts = [t for t in texts if t != best_text]
    if not other_texts:
        return 1.0

    try:
        # Один батчевый encode вместо N отдельных вызовов
        all_texts = [best_text] + other_texts
        embs = similarity_model.encode(all_texts, convert_to_numpy=True,
                                       show_progress_bar=False, batch_size=32)
        best_emb = embs[0]
        other_embs = embs[1:]

        # Векторизованное косинусное сходство
        dots = other_embs @ best_emb
        norms = np.linalg.norm(other_embs, axis=1) * np.linalg.norm(best_emb) + 1e-8
        similarities = (dots / norms + 1) / 2
        uncertainty = 1.0 - float(np.mean(similarities))
    except Exception:
        uncertainty = 1.0

    return uncertainty

def compute_cocoa_uncertainty(responses: List[Tuple[str, float]], 
                              similarity_model,
                              info_type: str = "SP") -> float:
    """
    U_CoCoA: Комбинированная метрика (Formula 10 в статье)
    
    Û_CoCoA(y*|x) = u(y*|x) × Û_cons(y*|x)
    """
    
    # Information-based uncertainty
    if info_type == "SP":
        info_uncertainty = compute_sequence_probability(responses)
    elif info_type == "PPL":
        info_uncertainty = compute_perplexity(responses)
    else:
        info_uncertainty = compute_sequence_probability(responses)
    
    # Consistency-based uncertainty
    cons_uncertainty = compute_consistency_uncertainty(similarity_model, responses)
    
    # Комбинация (произведение)
    cocoa_uncertainty = info_uncertainty * cons_uncertainty
    
    return cocoa_uncertainty

def compute_cocoa_light_uncertainty(response: Tuple[str, float], 
                                    embedding: np.ndarray,
                                    mlp_model: CoCoALightMLP,
                                    info_type: str = "SP") -> float:
    """
    U_CoCoA Light: С обученной MLP (Formula 11 в статье)
    
    Û_CoCoA^L(y*|x) = u(y*|x) × ĝ(e(y*|x))
    
    где ĝ - обученная MLP, e - эмбеддинги среднего слоя
    """
    
    text, log_prob = response
    
    # Information-based uncertainty (из одного greedy ответа)
    if info_type == "SP":
        probability = np.exp(log_prob)
        info_uncertainty = 1.0 - min(probability, 1.0)
    elif info_type == "PPL":
        perplexity = np.exp(-log_prob)
        info_uncertainty = min(perplexity / 100.0, 1.0)
    else:
        probability = np.exp(log_prob)
        info_uncertainty = 1.0 - min(probability, 1.0)
    
    # Consistency uncertainty из MLP (без sampling!)
    cons_uncertainty = mlp_model.predict(embedding)
    
    # Комбинация
    cocoa_light_uncertainty = info_uncertainty * cons_uncertainty
    
    return cocoa_light_uncertainty

# ============================================================================
# ОЦЕНКА КАЧЕСТВА
# ============================================================================

def compute_correctness(model_response: str, reference_answer: str, 
                        task_type: str) -> bool:
    """Проверяет правильность ответа"""
    
    if not model_response or not reference_answer:
        return False
    
    if task_type == "qa":
        ref_lower = reference_answer.lower()
        resp_lower = model_response.lower()
        ref_words = set(ref_lower.split())
        resp_words = set(resp_lower.split())
        overlap = len(ref_words & resp_words)
        return overlap >= max(1, len(ref_words) // 3)
    
    elif task_type == "summarization":
        ref_words = set(reference_answer.lower().split())
        resp_words = set(model_response.lower().split())
        overlap = len(ref_words & resp_words)
        return overlap >= max(5, len(ref_words) // 4)
    
    elif task_type == "translation":
        if len(model_response) < len(reference_answer) * 0.5:
            return False
        ref_words = set(reference_answer.lower().split())
        resp_words = set(model_response.lower().split())
        overlap = len(ref_words & resp_words)
        return overlap >= max(3, len(ref_words) // 4)
    
    return False

def compute_prediction_rejection_ratio(uncertainties: List[float], 
                                       correctness: List[bool],
                                       max_rejection: float = 0.5) -> float:
    """
    PRR: Prediction Rejection Ratio
    
    Вычисляется только до max_rejection (50% в статье)
    """
    
    if len(uncertainties) != len(correctness):
        return 0.0
    
    n = len(uncertainties)
    if n == 0:
        return 0.0
    
    sorted_indices = np.argsort(-np.array(uncertainties))
    base_accuracy = np.mean(correctness)
    
    k_steps = 10
    prr_values = []
    
    for k in range(1, k_steps + 1):
        rejection_rate = k / k_steps
        if rejection_rate > max_rejection:
            break
        
        num_rejected = int(n * rejection_rate)
        if num_rejected >= n:
            continue
        
        kept_indices = sorted_indices[num_rejected:]
        kept_correctness = [correctness[i] for i in kept_indices]
        
        if kept_correctness:
            accuracy_at_k = np.mean(kept_correctness)
            prr_values.append(accuracy_at_k - base_accuracy)
    
    prr = np.mean(prr_values) if prr_values else 0.0
    return prr

def compute_auroc(uncertainties: List[float], correctness: List[bool]) -> float:
    """AUROC: Area Under ROC Curve"""
    
    try:
        from sklearn.metrics import roc_auc_score
        labels = [1 if c else 0 for c in correctness]
        scores = [1 - u for u in uncertainties]
        auroc = roc_auc_score(labels, scores)
        return auroc
    except:
        return 0.5

# ============================================================================
# ОБУЧЕНИЕ COCOA LIGHT
# ============================================================================

def train_cocoa_light(model, tokenizer, similarity_model, train_data: List[Dict], 
                      config: Config) -> CoCoALightMLP:
    """
    Обучает CoCoA Light MLP на held-out set (без ground truth labels!)
    
    Согласно Appendix I:
    1. Для каждого примера генерируем multiple samples
    2. Вычисляем true consistency uncertainty
    3. Извлекаем эмбеддинги среднего слоя
    4. Обучаем MLP предсказывать consistency из эмбеддингов
    """
    
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ COCOA LIGHT MLP")
    print("=" * 80)
    
    embeddings_list = []
    targets_list = []
    
    for item in tqdm(train_data, desc="Подготовка training данных"):
        prompt = create_prompt(item)
        
        # Генерируем multiple samples для вычисления true consistency
        responses, embeddings = generate_responses_with_embeddings(
            model, tokenizer, prompt, 
            num_samples=config.NUM_SAMPLES,
            config=config,
            return_embeddings=True
        )
        
        # Фильтруем пустые ответы, сохраняя выравнивание с embeddings
        if embeddings:
            paired = [(r, e) for r, e in zip(responses, embeddings) if r[0].strip() and e is not None]
            if len(paired) < 2:
                continue
            responses = [x[0] for x in paired]
            embeddings = [x[1] for x in paired]
        else:
            responses = [(t, p) for t, p in responses if t.strip()]
            if len(responses) < 2:
                continue

        # Вычисляем true consistency uncertainty (target для MLP)
        true_cons_uncertainty = compute_consistency_uncertainty(similarity_model, responses)

        # Берём эмбеддинг из лучшего ответа (greedy-like)
        best_idx = np.argmax([r[1] for r in responses])
        best_embedding = embeddings[best_idx]
        
        embeddings_list.append(best_embedding)
        targets_list.append(true_cons_uncertainty)
    
    # Конвертируем в numpy arrays
    embeddings_array = np.array(embeddings_list)
    targets_array = np.array(targets_list)
    
    print(f"\nПодготовлено {len(embeddings_array)} примеров для обучения")
    
    # Создаём и обучаем MLP
    mlp = CoCoALightMLP(input_dim=embeddings_array.shape[1], hidden_dim=2048)
    mlp.train(embeddings_array, targets_array)
    
    return mlp

# ============================================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================================

def run_experiment():
    """Запускает полный эксперимент CoCoA"""
    
    config = Config()
    
    print("=" * 80)
    print("CoCoA Uncertainty Quantification Experiment")
    print("   Задачи: QA, Summarization, Machine Translation (NMT)")
    print("=" * 80)
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Устройство: {config.DEVICE}")
    print(f"4-bit квантование: {config.USE_4BIT}")
    print(f"Число samples (CoCoA): {config.NUM_SAMPLES}")
    print(f"Слой эмбеддингов (CoCoA Light): {config.EMBEDDING_LAYER}")
    print(f"Модель сходства: {config.SIMILARITY_MODEL}")
    print("=" * 80)
    
    # Загружаем модели
    llm_model, tokenizer = load_llm_model(config.MODEL_NAME, config.USE_4BIT)
    similarity_model = load_similarity_model(config.SIMILARITY_MODEL)
    
    # Обучаем CoCoA Light MLP (если нужно)
    mlp_model = None
    if config.RUN_COCOA_LIGHT:
        # Загружаем training data
        train_data = load_mini_dataset("TriviaQA", {
            "name": "trivia_qa", 
            "subset": "rc.wikipedia.nocontext",
            "size": config.LIGHT_TRAIN_SIZE, 
            "task_type": "qa"
        })
        
        # Обучаем MLP
        mlp_model = train_cocoa_light(
            llm_model, tokenizer, similarity_model, 
            train_data, config
        )
    
    # Результаты
    all_results = []
    
    for dataset_name, dataset_config in config.DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Датасет: {dataset_name} ({dataset_config['task_type'].upper()})")
        print(f"{'='*80}")
        
        # Загружаем данные
        data = load_mini_dataset(dataset_name, dataset_config)
        
        # Метрики
        uncertainties_methods = {
            "SequenceProb": [],
            "Perplexity": [],
            "Consistency": [],
            "CoCoA_SP": [],
            "CoCoA_PPL": [],
        }
        
        if config.RUN_COCOA_LIGHT and mlp_model is not None:
            uncertainties_methods["CoCoA_Light_SP"] = []
            uncertainties_methods["CoCoA_Light_PPL"] = []
        
        correctness_list = []
        
        # Обрабатываем каждый пример
        for i, item in enumerate(tqdm(data, desc=f"Обработка {dataset_name}")):
            
            # Формируем промпт
            prompt = create_prompt(item)
            
            # Генерируем ответы (для CoCoA)
            responses, embeddings = generate_responses_with_embeddings(
                llm_model, tokenizer, prompt, 
                config.NUM_SAMPLES, config,
                return_embeddings=config.RUN_COCOA_LIGHT
            )
            
            # Фильтруем пустые ответы, сохраняя выравнивание с embeddings
            if embeddings:
                paired = [(r, e) for r, e in zip(responses, embeddings) if r[0].strip() and e is not None]
                responses = [x[0] for x in paired]
                embeddings = [x[1] for x in paired]
            else:
                responses = [(t, p) for t, p in responses if t.strip()]

            if not responses:
                continue
            
            # Вычисляем неопределённости (CoCoA)
            if config.RUN_COCOA:
                u_sp = compute_sequence_probability(responses)
                u_ppl = compute_perplexity(responses)
                u_cons = compute_consistency_uncertainty(similarity_model, responses)
                u_cocoa_sp = compute_cocoa_uncertainty(responses, similarity_model, "SP")
                u_cocoa_ppl = compute_cocoa_uncertainty(responses, similarity_model, "PPL")
                
                uncertainties_methods["SequenceProb"].append(u_sp)
                uncertainties_methods["Perplexity"].append(u_ppl)
                uncertainties_methods["Consistency"].append(u_cons)
                uncertainties_methods["CoCoA_SP"].append(u_cocoa_sp)
                uncertainties_methods["CoCoA_PPL"].append(u_cocoa_ppl)
            
            # Вычисляем неопределённости (CoCoA Light)
            if config.RUN_COCOA_LIGHT and mlp_model is not None and embeddings:
                best_idx = np.argmax([r[1] for r in responses])
                best_response = responses[best_idx]
                best_embedding = embeddings[best_idx]
                
                u_light_sp = compute_cocoa_light_uncertainty(
                    best_response, best_embedding, mlp_model, "SP"
                )
                u_light_ppl = compute_cocoa_light_uncertainty(
                    best_response, best_embedding, mlp_model, "PPL"
                )
                
                uncertainties_methods["CoCoA_Light_SP"].append(u_light_sp)
                uncertainties_methods["CoCoA_Light_PPL"].append(u_light_ppl)
            
            # Проверяем правильность
            best_response = max(responses, key=lambda x: x[1])[0]
            is_correct = compute_correctness(best_response, item["reference"], item["task_type"])
            correctness_list.append(is_correct)
        
        # Вычисляем PRR для каждого метода
        prr_results = {}
        auroc_results = {}
        
        for method, uncertainties in uncertainties_methods.items():
            if uncertainties and len(uncertainties) == len(correctness_list):
                prr = compute_prediction_rejection_ratio(uncertainties, correctness_list)
                auroc = compute_auroc(uncertainties, correctness_list)
                prr_results[method] = prr
                auroc_results[method] = auroc
        
        # Сохраняем результаты
        result = {
            "Dataset": dataset_name,
            "Task": dataset_config["task_type"].upper(),
            "Samples": len(correctness_list),
            "Base_Accuracy": f"{np.mean(correctness_list):.3f}",
            "PRR_SequenceProb": f"{prr_results.get('SequenceProb', 0):.3f}",
            "PRR_Perplexity": f"{prr_results.get('Perplexity', 0):.3f}",
            "PRR_Consistency": f"{prr_results.get('Consistency', 0):.3f}",
            "PRR_CoCoA_SP": f"{prr_results.get('CoCoA_SP', 0):.3f}",
            "PRR_CoCoA_PPL": f"{prr_results.get('CoCoA_PPL', 0):.3f}",
        }
        
        if config.RUN_COCOA_LIGHT and mlp_model is not None:
            result["PRR_CoCoA_Light_SP"] = f"{prr_results.get('CoCoA_Light_SP', 0):.3f}"
            result["PRR_CoCoA_Light_PPL"] = f"{prr_results.get('CoCoA_Light_PPL', 0):.3f}"
            result["AUROC_CoCoA_Light"] = f"{auroc_results.get('CoCoA_Light_SP', 0):.3f}"
        
        result["AUROC_CoCoA"] = f"{auroc_results.get('CoCoA_SP', 0):.3f}"
        
        all_results.append(result)
        
        print(f"Обработано {len(correctness_list)} примеров")
        print(f"   Base Accuracy: {np.mean(correctness_list):.3f}")
        print(f"   PRR CoCoA_SP: {prr_results.get('CoCoA_SP', 0):.3f}")
        if config.RUN_COCOA_LIGHT and mlp_model is not None:
            print(f"   PRR CoCoA_Light_SP: {prr_results.get('CoCoA_Light_SP', 0):.3f}")
    
    # Создаём таблицу результатов
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📋 ТАБЛИЦА РЕЗУЛЬТАТОВ (Table 1)")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Сохраняем
    df.to_csv("cocoa_results_corrected.csv", index=False)
    print(f"\nРезультаты сохранены в cocoa_results_corrected.csv")
    
    try:
        df.to_excel("cocoa_results_corrected.xlsx", index=False)
        print(f"Результаты сохранены в cocoa_results_corrected.xlsx")
    except:
        print("Не удалось сохранить в Excel (pip install openpyxl)")
    
    return df

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    results_df = run_experiment()
