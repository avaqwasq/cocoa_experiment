@echo off
REM setup.bat - Автоматическая настройка окружения для CoCoA (Windows)

echo ==========================================
echo CoCoA Experiment Setup
echo ==========================================

REM 1. Проверка Python
echo Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.9+
    pause
    exit /b 1
)
echo ✅ Python найден

REM 2. Создание виртуального окружения
echo Создание виртуального окружения...
if exist "cocoa_env" (
    echo ⚠️  cocoa_env уже существует
    set /p OVERWRITE="Пересоздать? (y/n): "
    if /i "%OVERWRITE%"=="y" (
        rmdir /s /q cocoa_env
        python -m venv cocoa_env
        echo ✅ cocoa_env пересоздано
    )
) else (
    python -m venv cocoa_env
    echo ✅ cocoa_env создано
)

REM 3. Активация окружения
echo Активация виртуального окружения...
call cocoa_env\Scripts\activate.bat

REM 4. Обновление pip
echo Обновление pip...
python -m pip install --upgrade pip

REM 5. Установка зависимостей
echo Установка зависимостей из requirements.txt...
pip install -r requirements.txt

REM 6. Проверка установки
echo Проверка установки...
python -c "import torch; import transformers; import sentence_transformers"
if errorlevel 1 (
    echo ❌ Ошибка проверки зависимостей!
    pause
    exit /b 1
)
echo ✅ Все зависимости установлены успешно!

echo.
echo ==========================================
echo ✅ Настройка завершена!
echo ==========================================
echo.
echo Для запуска эксперимента:
echo   cocoa_env\Scripts\activate
echo   python cocoa_experiment_full.py
echo.
echo Для деактивации окружения:
echo   deactivate
echo.
pause