# cocoa_experiment
Проведение на основе статьи "CoCoA: A Minimum Bayes Risk Framework Bridging Confidence and Consistency for Uncertainty Quantification in LLMs" эсперимента по вычислению значений prr на основе разных метрик на малой модели Qwen2.5-1.5B-Instruct.

Цель эксперимента: добиться повышения качества значений prr на основе COCOA и COCOA_light на модели Qwen2.5-1.5B-Instruct

В оригинальной статье были представлены результаты на БЯМ, я же хочу показать, что метод будет работать и на малых моделях.

## 🚀 Быстрый старт

--Linux/Mac--
```bash
# 1. Клонирование/скачивание
git clone <repo_url>
cd cocoa_experiment

# 2. Автоматическая настройка
chmod +x setup.sh
./setup.sh

# 3. Запуск
source cocoa_env/bin/activate
python cocoa_experiment_full.py

--Windows--
REM 1. Настройка
setup.bat

REM 2. Запуск
cocoa_env\Scripts\activate
python cocoa_experiment_full.py

--Make (Linux/Mac)--
make setup    # Настройка
make run      # Запуск
make clean    # Очистка
