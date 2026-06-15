@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Chest X-ray Classifier  -  app launcher
echo ============================================================
echo.

REM --- Find a TensorFlow-compatible Python (3.12, 3.11, or 3.10) -------------
REM TensorFlow does NOT support Python 3.13+, so we must not use a newer one.
set "PYV="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYV (
        py -%%V -c "import sys" >nul 2>&1 && set "PYV=%%V"
    )
)

if not defined PYV (
    echo ERROR: Could not find Python 3.10, 3.11, or 3.12.
    echo TensorFlow does not support Python 3.13 or newer, so the app needs one of those.
    echo Install Python 3.12 from https://www.python.org/downloads/ and run this again.
    echo.
    pause
    exit /b 1
)
echo Using Python !PYV!.

REM --- Create the environment once ------------------------------------------
if not exist ".venv-app\Scripts\python.exe" (
    echo Creating a local environment. First run only, please wait...
    py -!PYV! -m venv .venv-app
)
set "VPY=.venv-app\Scripts\python.exe"

REM --- Install the app's packages -------------------------------------------
echo Installing packages (the first run downloads TensorFlow, this takes a few minutes)...
"%VPY%" -m pip install --upgrade pip >nul
"%VPY%" -m pip install -r cxr_app\requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: installing the packages failed. See the messages above.
    pause
    exit /b 1
)

REM Skip Streamlit's first-run "enter your email" prompt so it launches unattended.
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
if exist "%USERPROFILE%\.streamlit\credentials.toml" goto :launch
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
> "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
>> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
:launch

echo.
echo ============================================================
echo   Starting the app. Your browser should open automatically.
echo   If it does not, open this address yourself:
echo        http://localhost:8501
echo   The first load takes about 15-20 seconds while the model loads.
echo   KEEP THIS WINDOW OPEN while you use the app. Close it to stop.
echo ============================================================
echo.
"%VPY%" -m streamlit run cxr_app\app.py

pause
