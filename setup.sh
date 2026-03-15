#!/bin/bash
# setup.sh - Автоматическая настройка окружения для CoCoA

set -e  # Остановиться при ошибке

echo "=========================================="
echo "🚀 CoCoA Experiment Setup"
echo "=========================================="

# 1. Проверка Python
echo "📋 Проверка Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден! Установите Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION найден"

# 2. Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
if [ -d "cocoa_env" ]; then
    echo "⚠️  cocoa_env уже существует"
    read -p "Пересоздать? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf cocoa_env
        python3 -m venv cocoa_env
        echo "✅ cocoa_env пересоздано"
    fi
else
    python3 -m venv cocoa_env
    echo "✅ cocoa_env создано"
fi

# 3. Активация окружения
echo "🔌 Активация виртуального окружения..."
source cocoa_env/bin/activate

# 4. Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip

# 5. Установка зависимостей
echo "📦 Установка зависимостей из requirements.txt..."
pip install -r requirements.txt

# 6. Проверка установки
echo "🔍 Проверка установки..."
python -c "import torch; import transformers; import sentence_transformers" && \
    echo "✅ Все зависимости установлены успешно!" || \
    (echo "❌ Ошибка проверки зависимостей!" && exit 1)

# 7. Информация
echo ""
echo "=========================================="
echo "✅ Настройка завершена!"
echo "=========================================="
echo ""
echo "Для запуска эксперимента:"
echo "  source cocoa_env/bin/activate"
echo "  python cocoa_experiment_full.py"
echo ""
echo "Для деактивации окружения:"
echo "  deactivate"
echo ""