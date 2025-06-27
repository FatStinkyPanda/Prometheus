@echo off
REM ============================================================================
REM == Prometheus Consciousness System - Windows Automated Launcher
REM ==
REM == This script automates the entire setup and launch process:
REM == 1. Checks for a system-wide Python installation.
REM == 2. Runs the setup script to create/verify the virtual environment
REM ==    and install all dependencies.
REM == 3. Activates the virtual environment.
REM == 4. Runs the main application launcher within the activated environment.
REM ==
REM == To use, simply double-click this file or run "start_prometheus.bat"
REM == from your command prompt.
REM ============================================================================
echo.
echo ============================================================
echo       Prometheus Consciousness System - Automated Start
echo ============================================================
echo.

REM --- Step 1: Check for Python ---
echo [1/4] Checking for system Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system's PATH.
    echo Please install Python 3.9+ and ensure it is added to your PATH.
    pause
    exit /b 1
)
echo [SUCCESS] Python found.
echo.

REM --- Step 2: Run the Initial Setup ---
echo [2/4] Executing the setup and dependency installation script...
echo This may take several minutes on the first run as models are downloaded.
echo.
python run_prometheus.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The initial setup script failed. Please review the errors above.
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Initial setup script completed.
echo.


REM --- Step 3: Activate the Virtual Environment ---
REM The previous script exits after creating the venv, so now we activate it.
set "VENV_PATH=%~dp0venv\Scripts\activate.bat"
if not exist "%VENV_PATH%" (
    echo [ERROR] Virtual environment activation script not found at:
    echo %VENV_PATH%
    echo The setup may have failed.
    pause
    exit /b 1
)
echo [3/4] Activating the virtual environment...
call "%VENV_PATH%"
echo [SUCCESS] Virtual environment is now active.
echo.


REM --- Step 4: Launch the Main Application ---
echo [4/4] Starting the Prometheus application...
echo The GUI window will now appear.
echo.
python run_prometheus.py

echo.
echo ============================================================
echo          Prometheus application has been closed.
echo ============================================================
echo.
pause