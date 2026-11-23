@echo off
title KALANTAR Railway Intelligence Server
cd /d "%~dp0"
echo Starting KALANTAR Server on http://127.0.0.1:8050 ...
echo Press Ctrl+C to stop the server anytime.
echo.
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8050 --reload
pause
