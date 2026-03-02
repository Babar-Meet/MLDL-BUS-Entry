@echo off
setlocal

:: Change to script directory
cd /d %~dp0

echo ================================================
echo   INDIAN NUMBER PLATE DETECTION SYSTEM
echo   Fully Automated
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
    echo Please run setup.bat first to create the environment
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo Please run setup.bat to recreate the environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

:: Check for arguments
if "%~1"=="" (
    echo [MODE] Auto mode (webcam -> video -> demo)
    echo.
    echo Starting automatic detection...
    echo Press Ctrl+C to stop
    echo.
    python main.py --auto
    goto end
)

:: Check for explicit flags
if /i "%~1"=="--auto" (
    echo [MODE] Auto mode (webcam -> video -> demo)
    echo.
    echo Starting automatic detection...
    echo Press Ctrl+C to stop
    echo.
    python main.py --auto
    goto end
)

if /i "%~1"=="--webcam" (
    echo [MODE] Webcam mode
    echo.
    echo Starting webcam detection...
    echo Press Ctrl+C to stop
    echo.
    python main.py --webcam
    goto end
)

if /i "%~1"=="--image" (
    if "%~2"=="" (
        echo [ERROR] No image file specified
        echo Usage: run.bat --image image.jpg
        pause
        exit /b 1
    )
    echo [MODE] Image mode
    echo.
    echo Processing image: %~2
    echo.
    python main.py --image "%~2"
    goto end
)

if /i "%~1"=="--video" (
    if "%~2"=="" (
        echo [ERROR] No video file specified
        echo Usage: run.bat --video video.mp4
        pause
        exit /b 1
    )
    echo [MODE] Video mode
    echo.
    echo Processing video: %~2
    echo Press Ctrl+C to stop
    echo.
    python main.py --video "%~2"
    goto end
)

:: Check if first argument is a file
set INPUT_FILE=%~1

:: Check if file exists
if not exist "%INPUT_FILE%" (
    echo [ERROR] File not found: %INPUT_FILE%
    pause
    exit /b 1
)

:: Get file extension
for %%A in ("%INPUT_FILE%") do set EXTENSION=%%~xA

:: Check if it's an image or video
if /i "%EXTENSION%"==".jpg" goto image_mode
if /i "%EXTENSION%"==".jpeg" goto image_mode
if /i "%EXTENSION%"==".png" goto image_mode
if /i "%EXTENSION%"==".bmp" goto image_mode
if /i "%EXTENSION%"==".mp4" goto video_mode
if /i "%EXTENSION%"==".avi" goto video_mode
if /i "%EXTENSION%"==".mov" goto video_mode
if /i "%EXTENSION%"==".mkv" goto video_mode

echo [ERROR] Unsupported file format: %EXTENSION%
echo Supported formats: jpg, jpeg, png, bmp, mp4, avi, mov, mkv
pause
exit /b 1

:image_mode
    echo [MODE] Image mode
    echo.
    echo Processing image: %INPUT_FILE%
    echo.
    python main.py --image "%INPUT_FILE%"
    goto end

:video_mode
    echo [MODE] Video mode
    echo.
    echo Processing video: %INPUT_FILE%
    echo Press Ctrl+C to stop early
    echo.
    python main.py --video "%INPUT_FILE%"
    goto end

:end
if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred during processing
) else (
    echo.
    echo [SUCCESS] Processing complete!
    echo Check the 'output' folder for results
    echo Check the 'database' folder for logged detections
)

:: Keep console open
pause
