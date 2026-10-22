.PHONY: run baseline docker up clean
run:
	python src/run_experiment.py
baseline:
	python src/train_ml_baseline.py
docker:
	docker build -t mlspec-experiment:v1 .
up:
	docker compose up --build
clean:
	rm -rf mlruns mlflow.db artifacts/*.json
