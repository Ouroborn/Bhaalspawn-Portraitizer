@echo off
REM Скрипт установки зависимостей для Baldur's Gate EE Portrait Converter

echo 🚀 Установка зависимостей для Baldur's Gate EE Portrait Converter...
pip install -r requirements.txt

echo.
echo ✅ Установка завершена!
echo.
echo 📖 Использование:
echo   python bg_portrait_converter.py --input C:\path\to\images
echo   python bg_portrait_converter.py -i C:\path\to\images -o C:\path\to\output
echo.
echo Если не указать параметры, программа попросит их интерактивно.
pause
