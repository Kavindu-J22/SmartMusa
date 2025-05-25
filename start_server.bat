@echo off
echo Starting SmartMusa Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Check if model exists
if not exist banana_price_model.pkl (
    echo Model not found. Training model first...
    python train_model.py
)

REM Start the Flask server
echo.
echo Starting the Flask server...
python app.py

pause