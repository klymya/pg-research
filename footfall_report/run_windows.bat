@echo off
REM Start the footfall report app on Windows.
REM Creates the virtual environment and installs dependencies on first run.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv || python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
    echo Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting the app - it opens in your browser. Close this window to stop it.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
