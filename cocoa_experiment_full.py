#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoCoA Uncertainty Quantification Experiment
Поддержка QA, SUMMARIZATION и MACHINE TRANSLATION (NMT)
Для слабых ноутбуков (4-bit квантование, мини-датасеты)
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Конфигурация эксперимента"""
    
    # Модель (мультиязычная для NMT!)
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  
    # Альтернативы для NMT: "google/gemma-2-2b-it", "microsoft/Phi-3.5-mini-instruct"
    
    # Параметры генерации
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.7
    TOP_P = 0.9
    NUM_SAMPLES = 5  # Уменьшено для ноутбука
    
    # Датасеты (3 типа задач)
    DATASETS = {
        # QA Tasks
        "TriviaQA": {"name": "trivia_qa", "subset": "rc.wikipedia.nocontext", 
                     "size": 300, "task_type": "qa"},
        "CoQA": {"name": "coqa", "subset": None, 
                 "size": 300, "task_type": "qa"},
        
        # Summarization Task
        "XSUM": {"name": "xsum", "subset": None, 
                 "size": 300, "task_type": "summarization"},
        
        # Machine Translation Task (NMT)
        "WMT19_EN_DE": {"name": "wmt19", "subset": "de-en", 
                        "size": 300, "task_type": "translation"},
        # IWSLT легче чем WMT, рекомендую для слабых ноутбуков
        "IWSLT_EN_DE": {"name": "iwslt2017", "subset": "en-de", 
                        "size": 300, "task_type": "translation"},
    }
    
    # Функция сходства (мультиязычная для NMT!)
    # LaBSE поддерживает 100+ языков, лучше для перевода
    SIMILARITY_MODEL = "sentence-transformers/LaBSE"  # Или "all-MiniLM-L6-v2"
    
    # Устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4-bit квантование
    USE_4BIT = True
    COMPUTE_DTYPE = torch.float16

# ============================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================================

def load_llm_model(model_name: str, use_4bit: bool = True):
    """Загружает языковую модель с квантованием"""
    
    print(f"Загрузка модели: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if use_4bit and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
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
    """Загружает мини-версию датасета для QA, SUM или NMT"""
    
    task_type = dataset_config["task_type"]
    size = dataset_config["size"]
    
    print(f"🔄 Загрузка датасета: {dataset_name} ({task_type}, {size} примеров)")
    
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
        if dataset_name == "WMT19_EN_DE":
            # WMT19 de-en (немецко-английский)
            dataset = load_dataset("wmt19", "de-en", split="test")
            for i, item in enumerate(dataset):
                if i >= size:
                    break
                data.append({
                    "input": item["translation"]["de"],  # Немецкий (источник)
                    "reference": item["translation"]["en"],  # Английский (цель)
                    "task_type": "translation",
                    "src_lang": "de",
                    "tgt_lang": "en"
                })
        
        elif dataset_name == "IWSLT_EN_DE":
            # IWSLT2017 en-de (англо-немецкий) - легче!
            dataset = load_dataset("iwslt2017", "en-de", split="test")
            for i, item in enumerate(dataset):
                if i >= size:
                    break
                data.append({
                    "input": item["translation"]["en"],  # Английский (источник)
                    "reference": item["translation"]["de"],  # Немецкий (цель)
                    "task_type": "translation",
                    "src_lang": "en",
                    "tgt_lang": "de"
                })
    
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")
    
    print(f"Загружено {len(data)} примеров")
    return data

# ============================================================================
# ФОРМИРОВАНИЕ ПРОМПТОВ
# ============================================================================

def create_prompt(item: Dict, model_name: str = "Qwen") -> str:
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
        
        # Явно указываем языки для лучшего качества перевода
        lang_names = {
            "en": "English", "de": "German", "fr": "French", 
            "es": "Spanish", "ru": "Russian", "zh": "Chinese"
        }
        
        prompt = f"Translate the following text from {lang_names.get(src_lang, src_lang)} to {lang_names.get(tgt_lang, tgt_lang)}:\n\n{input_text}\n\nTranslation:"
    
    else:
        prompt = input_text
    
    return prompt

# ============================================================================
# ГЕНЕРАЦИЯ ОТВЕТОВ
# ============================================================================

def generate_responses(model, tokenizer, prompt: str, num_samples: int, 
                       config: Config) -> List[Tuple[str, float]]:
    """Генерирует multiple samples и возвращает (текст, log_probability)"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                       max_length=512).to(config.DEVICE)
    input_length = inputs["input_ids"].shape[1]
    
    responses = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Декодируем ответ
        generated_ids = outputs.sequences[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Вычисляем log probability (confidence)
        if hasattr(outputs, 'scores') and outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                token_id = outputs.sequences[0][input_length + i]
                log_prob = torch.log_softmax(score[0], dim=0)[token_id].item()
                log_probs.append(log_prob)
            avg_log_prob = np.mean(log_probs) if log_probs else 0.0
        else:
            avg_log_prob = 0.0
        
        responses.append((text, avg_log_prob))
    
    return responses

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
        similarity = (similarity + 1) / 2  # Нормализуем от 0 до 1
        return float(similarity)
    except:
        return 0.0

def compute_consistency_uncertainty(similarity_model, responses: List[Tuple[str, float]]) -> float:
    """U_cons: Consistency-based uncertainty"""
    
    if len(responses) < 2:
        return 1.0
    
    texts = [r[0] for r in responses]
    
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = compute_semantic_similarity(similarity_model, texts[i], texts[j])
            similarities.append(sim)
    
    if not similarities:
        return 1.0
    
    avg_similarity = np.mean(similarities)
    uncertainty = 1.0 - avg_similarity
    return uncertainty

def compute_cocoa_uncertainty(responses: List[Tuple[str, float]], 
                              similarity_model) -> float:
    """U_CoCoA: Комбинированная метрика (confidence × consistency)"""
    
    info_uncertainty = compute_sequence_probability(responses)
    cons_uncertainty = compute_consistency_uncertainty(similarity_model, responses)
    cocoa_uncertainty = info_uncertainty * cons_uncertainty
    
    return cocoa_uncertainty

def compute_cocoa_light_uncertainty(responses: List[Tuple[str, float]], 
                                    similarity_model,
                                    learned_weight: float = 0.5) -> float:
    """U_CoCoA Light: Аппроксимация"""
    
    approx_responses = responses[:3] if len(responses) >= 3 else responses
    
    info_uncertainty = compute_sequence_probability(approx_responses)
    cons_uncertainty = compute_consistency_uncertainty(similarity_model, approx_responses)
    
    cocoa_light = info_uncertainty * (learned_weight * cons_uncertainty + 
                                       (1 - learned_weight) * info_uncertainty)
    
    return cocoa_light

# ============================================================================
# ОЦЕНКА КАЧЕСТВА
# ============================================================================

def compute_correctness(model_response: str, reference_answer: str, 
                        task_type: str) -> bool:
    """Проверяет правильность ответа в зависимости от задачи"""
    
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
        # Для суммаризации используем overlap слов
        ref_words = set(reference_answer.lower().split())
        resp_words = set(model_response.lower().split())
        overlap = len(ref_words & resp_words)
        return overlap >= max(5, len(ref_words) // 4)
    
    elif task_type == "translation":
        # Для перевода проверяем длину и overlap
        if len(model_response) < len(reference_answer) * 0.5:
            return False
        ref_words = set(reference_answer.lower().split())
        resp_words = set(model_response.lower().split())
        overlap = len(ref_words & resp_words)
        return overlap >= max(3, len(ref_words) // 4)
    
    return False

def compute_prediction_rejection_ratio(uncertainties: List[float], 
                                       correctness: List[bool]) -> float:
    """PRR: Prediction Rejection Ratio"""
    
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
        num_rejected = int(n * k / k_steps)
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
    print(f"Число samples: {config.NUM_SAMPLES}")
    print(f"Модель сходства: {config.SIMILARITY_MODEL}")
    print("=" * 80)
    
    # Загружаем модели
    llm_model, tokenizer = load_llm_model(config.MODEL_NAME, config.USE_4BIT)
    similarity_model = load_similarity_model(config.SIMILARITY_MODEL)
    
    # Результаты
    all_results = []
    
    for dataset_name, dataset_config in config.DATASETS.items():
        print(f"\n{'='*80}")
        print(f"📊 Датасет: {dataset_name} ({dataset_config['task_type'].upper()})")
        print(f"{'='*80}")
        
        # Загружаем данные
        data = load_mini_dataset(dataset_name, dataset_config)
        
        # Метрики
        uncertainties_methods = {
            "SequenceProb": [],
            "Perplexity": [],
            "Consistency": [],
            "CoCoA": [],
            "CoCoA_Light": [],
        }
        correctness_list = []
        
        # Обрабатываем каждый пример
        for i, item in enumerate(tqdm(data, desc=f"Обработка {dataset_name}")):
            
            # Формируем промпт
            prompt = create_prompt(item, config.MODEL_NAME)
            
            # Генерируем ответы
            responses = generate_responses(
                llm_model, tokenizer, prompt, 
                config.NUM_SAMPLES, config
            )
            
            # Фильтруем пустые ответы
            responses = [(t, p) for t, p in responses if t.strip()]
            
            if not responses:
                continue
            
            # Вычисляем неопределённости
            u_sp = compute_sequence_probability(responses)
            u_ppl = compute_perplexity(responses)
            u_cons = compute_consistency_uncertainty(similarity_model, responses)
            u_cocoa = compute_cocoa_uncertainty(responses, similarity_model)
            u_cocoa_light = compute_cocoa_light_uncertainty(responses, similarity_model)
            
            uncertainties_methods["SequenceProb"].append(u_sp)
            uncertainties_methods["Perplexity"].append(u_ppl)
            uncertainties_methods["Consistency"].append(u_cons)
            uncertainties_methods["CoCoA"].append(u_cocoa)
            uncertainties_methods["CoCoA_Light"].append(u_cocoa_light)
            
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
        all_results.append({
            "Dataset": dataset_name,
            "Task": dataset_config["task_type"].upper(),
            "Samples": len(correctness_list),
            "Base_Accuracy": f"{np.mean(correctness_list):.3f}",
            "PRR_SequenceProb": f"{prr_results.get('SequenceProb', 0):.3f}",
            "PRR_Perplexity": f"{prr_results.get('Perplexity', 0):.3f}",
            "PRR_Consistency": f"{prr_results.get('Consistency', 0):.3f}",
            "PRR_CoCoA": f"{prr_results.get('CoCoA', 0):.3f}",
            "PRR_CoCoA_Light": f"{prr_results.get('CoCoA_Light', 0):.3f}",
            "AUROC_CoCoA": f"{auroc_results.get('CoCoA', 0):.3f}",
        })
        
        print(f"Обработано {len(correctness_list)} примеров")
        print(f"   Base Accuracy: {np.mean(correctness_list):.3f}")
        print(f"   PRR CoCoA: {prr_results.get('CoCoA', 0):.3f}")
        print(f"   AUROC CoCoA: {auroc_results.get('CoCoA', 0):.3f}")
    
    # Создаём таблицу результатов
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📋 ТАБЛИЦА РЕЗУЛЬТАТОВ (Table 1)")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Сохраняем
    df.to_csv("cocoa_results_full.csv", index=False)
    print(f"\nРезультаты сохранены в cocoa_results_full.csv")
    
    try:
        df.to_excel("cocoa_results_full.xlsx", index=False)
        print(f"Результаты сохранены в cocoa_results_full.xlsx")
    except:
        print("Не удалось сохранить в Excel (установите openpyxl: pip install openpyxl)")
    
    return df

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    results_df = run_experiment()