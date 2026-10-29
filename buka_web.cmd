@echo off
title Web Benchmark Komputasi Paralel
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python proyek tidak ditemukan.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo File app.py tidak ditemukan.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run "app.py" --server.address 127.0.0.1 --server.port 8501

pause