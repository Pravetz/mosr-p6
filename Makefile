.PHONY: run baseline mlflow mlflow-run mlflow-baseline docker up down clean clean-docker

MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
MLFLOW_STATE_DIR ?= .mlflow

run:
	python src/run_experiment.py
baseline:
	python src/train_ml_baseline.py
mlflow:
	mkdir -p $(MLFLOW_STATE_DIR)/artifacts
	mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///$(MLFLOW_STATE_DIR)/mlflow.db \
		--artifacts-destination $(MLFLOW_STATE_DIR)/artifacts
mlflow-run:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) python src/run_experiment.py --use-mlflow
mlflow-baseline:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) python src/train_ml_baseline.py --use-mlflow
docker:
	docker build -t mlspec-experiment:v1 .
up:
	docker compose up --build
down:
	docker compose down
clean:
	rm -rf .mlflow mlruns mlflow.db
	rm -f artifacts/metrics.json
clean-docker:
	docker compose down --volumes
