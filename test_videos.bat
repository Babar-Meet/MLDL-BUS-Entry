@echo off
setlocal

:: Change to script directory
cd /d %~dp0

echo ================================================
echo   NUMBER PLATE DETECTION EVALUATION
echo   Automated Testing System
echo ================================================
echo.
echo Running from: %CD%
echo.

:: Check if virtual environment exists and is valid
set VENV_VALID=0
if exist "venv" (
    if exist "venv\Scripts\python.exe" (
        set VENV_VALID=1
    )
)

if "%VENV_VALID%"=="0" (
    echo [ERROR] Virtual environment not found or incomplete
    echo Please run setup.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

:: Run evaluation
python evaluate.py

:: Keep console open
pause
