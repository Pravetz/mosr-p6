.PHONY: data run baseline reproduce analyze notebook test mlflow mlflow-run mlflow-baseline docker up down clean clean-docker

MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
MLFLOW_STATE_DIR ?= .mlflow
DOCKER ?= docker

data:
	python src/generate_benchmark.py
run:
	python src/run_experiment.py
baseline:
	python src/train_ml_baseline.py
reproduce:
	python src/check_reproducibility.py
analyze:
	python src/analyze_pr5.py
notebook:
	jupyter lab notebooks/pr5_statistical_analysis.ipynb
test:
	PYTHONPATH=src python -m unittest discover -s tests -v
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
	$(DOCKER) build -t mlspec-experiment:v1 .
up:
	GIT_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) $(DOCKER) compose up --build
down:
	$(DOCKER) compose down
clean:
	rm -rf .mlflow mlruns mlflow.db
	rm -f artifacts/metrics.json
clean-docker:
	$(DOCKER) compose down --volumes
