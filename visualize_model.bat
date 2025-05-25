@echo off
echo SmartMusa - Model Visualization Tool
echo -------------------------------------
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Check if model exists
if not exist banana_price_model.pkl (
    echo Model not found. Training model first...
    python train_model.py
) else (
    echo Model found. Generating visualizations...
    python visualize_model.py
)

echo.
echo Press any key to exit...
pause > nul