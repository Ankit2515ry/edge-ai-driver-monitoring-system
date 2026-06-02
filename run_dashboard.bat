@echo off
echo ===================================================
echo   Starting Edge AI Driver Monitoring Dashboard...
echo ===================================================

:: Activate the virtual environment
call env_stable\Scripts\activate

:: Run the Python script
python dashboard_ai.py

:: Keep the window open if the script crashes so you can read the error
pause