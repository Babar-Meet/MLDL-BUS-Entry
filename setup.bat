@echo off
setlocal enabledelayedexpansion

:: Change to script directory
cd /d %~dp0

echo ================================================
echo   INDIAN NUMBER PLATE DETECTION SYSTEM SETUP
echo   Fully Automated
echo ================================================
echo.
echo Running from: %CD%
echo.

:: Check if Python is installed
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python is installed

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo     Python version: %PYTHON_VERSION%

:: Create or fix virtual environment
echo.
echo [2/8] Creating virtual environment...

:: Check if venv exists and is valid
set VENV_VALID=0
if exist "venv" (
    if exist "venv\Scripts\python.exe" (
        set VENV_VALID=1
    ) else (
        echo     Removing incomplete venv...
        rmdir /s /q "venv" 2>nul
    )
)

if "%VENV_VALID%"=="0" (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo     Virtual environment already valid
)

:: Activate virtual environment
echo.
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo     Attempting to recreate...
    rmdir /s /q "venv" 2>nul
    python -m venv venv
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Failed to create/activate virtual environment
        pause
        exit /b 1
    )
)
echo [OK] Virtual environment activated

:: Install dependencies
echo.
echo [4/8] Installing dependencies...
pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Could not upgrade pip, continuing...
)

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

:: Create necessary folders
echo.
echo [5/8] Creating project folders...
set FOLDERS=models input output database logs snapshots
for %%F in (%FOLDERS%) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo     Created: %%F
    ) else (
        echo     Exists: %%F
    )
)
echo [OK] Folders created

:: Initialize database
echo.
echo [6/8] Setting up database...
python database\init_db.py
if errorlevel 1 (
    echo [ERROR] Database initialization failed
    echo.
    echo Please check if SQLite is available and try again
    pause
    exit /b 1
)
echo [OK] Database initialized

:: Download YOLO model
echo.
echo [7/8] Setting up YOLO model...
set MODEL_FILE=models\yolov8n.pt
if exist "%MODEL_FILE%" (
    echo     Model already exists: %MODEL_FILE%
) else (
    echo     Downloading YOLOv8n model...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded')" 2>nul
    if errorlevel 1 (
        echo [WARNING] Could not download model automatically
        echo Will download on first run
    ) else (
        :: Move model to models folder if it was downloaded
        if exist "yolov8n.pt" (
            move /y "yolov8n.pt" "%MODEL_FILE%" >nul 2>&1
            if exist "%MODEL_FILE%" (
                echo [OK] Model saved to %MODEL_FILE%
            )
        ) else (
            echo [OK] Model available from cache
        )
    )
)

:: Verify setup
echo.
echo [8/8] Verifying setup...
echo     Checking required folders...
set ALL_OK=1
for %%F in (models input output database snapshots) do (
    if not exist "%%F" (
        echo [ERROR] Missing folder: %%F
        set ALL_OK=0
    )
)
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment broken
    set ALL_OK=0
)

if "%ALL_OK%"=="0" (
    echo [ERROR] Setup verification failed
    pause
    exit /b 1
)
echo [OK] Setup verified

echo.
echo ================================================
echo   SETUP COMPLETE!
echo ================================================
echo.
echo To run the system:
echo   run.bat              - Auto mode (webcam -> video -> demo)
echo   run.bat --webcam    - Webcam mode
echo   run.bat image.jpg   - Image mode
echo   run.bat video.mp4   - Video mode
echo.
echo The system will automatically:
echo   - Use webcam if available
echo   - Fall back to input folder video/image
echo   - Generate demo image if nothing available
echo.
pause
