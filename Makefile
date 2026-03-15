# Makefile - Управление проектом CoCoA

.PHONY: setup run clean test help

# Установка окружения
setup:
	@echo "🚀 Настройка окружения..."
	@if [ ! -d "cocoa_env" ]; then \
		python3 -m venv cocoa_env; \
	fi
	@source cocoa_env/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
	@echo "✅ Готово!"

# Запуск эксперимента
run:
	@echo "🚀 Запуск CoCoA эксперимента..."
	@source cocoa_env/bin/activate && \
	python cocoa_experiment_full.py

# Запуск с конкретным датасетом
run-qa:
	@source cocoa_env/bin/activate && \
	python cocoa_experiment_full.py --datasets triviaqa coqa

run-sum:
	@source cocoa_env/bin/activate && \
	python cocoa_experiment_full.py --datasets xsum

run-nmt:
	@source cocoa_env/bin/activate && \
	python cocoa_experiment_full.py --datasets iwslt_en_de

# Очистка кэша
clean:
	@echo "🧹 Очистка кэша..."
	rm -rf .cache/huggingface
	rm -rf __pycache__
	rm -rf *.pyc
	rm -f *.csv *.xlsx
	@echo "✅ Очистка завершена!"

# Тестирование установки
test:
	@source cocoa_env/bin/activate && \
	python -c "import torch; import transformers; print('✅ Все работает!')"

# Помощь
help:
	@echo "CoCoA Experiment - Доступные команды:"
	@echo "  make setup   - Настройка окружения"
	@echo "  make run     - Запуск полного эксперимента"
	@echo "  make run-qa  - Запуск только QA задач"
	@echo "  make run-sum - Запуск только суммаризации"
	@echo "  make run-nmt - Запуск только перевода"
	@echo "  make clean   - Очистка кэша и результатов"
	@echo "  make test    - Проверка установки"
	@echo "  make help    - Показать эту справку"