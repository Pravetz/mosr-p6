@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\generate_benchmark.py
python src\run_experiment.py --seed 42
python src\train_ml_baseline.py --seed 42
echo.
echo Local run completed. To use MLflow, run: mlflow ui --port 5000
pause
