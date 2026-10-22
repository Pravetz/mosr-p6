# MLSpec methodological-validity reproducible artifact

This repository accompanies the study **"Method of formal specification and preventive verification of methodological validity of machine-learning experiments"**.

## What is included

- `data/benchmark_specs_pr4.csv` - controlled benchmark of 225 ML experiment specifications.
- `src/verifier.py` - rule-based preventive verifier.
- `src/run_experiment.py` - deterministic evaluation with optional MLflow logging.
- `src/train_ml_baseline.py` - deterministic ML baseline (Logistic Regression) for the benchmark labels.
- `Dockerfile` and `docker-compose.yml` - isolated environment and MLflow service.
- `data/benchmark_specs_pr4.csv.dvc` and `.dvc/config` - DVC metadata scaffold for the dataset.
- `zenodo.json` - metadata draft for publishing the research artifact.

## Local run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
python src/train_ml_baseline.py
```

## MLflow run

```bash
mlflow ui --host 127.0.0.1 --port 5000
python src/run_experiment.py --use-mlflow
python src/train_ml_baseline.py --use-mlflow
```

Open `http://localhost:5000`.

## Docker

```bash
docker build -t mlspec-experiment:v1 .
docker run --rm mlspec-experiment:v1
```

Or run the experiment and the MLflow server together:

```bash
docker compose up --build
```

## DVC

If DVC is installed, initialize/refresh the dataset pointer:

```bash
dvc init
dvc add data/benchmark_specs_pr4.csv
git add data/benchmark_specs_pr4.csv.dvc data/.gitignore .dvc/config
git commit -m "Track benchmark dataset with DVC"
```

A DVC-compatible pointer generated from the current dataset checksum is already included, but refreshing it with the installed DVC version is recommended.

## Reproducibility check

Run three times with the same seed:

```bash
python src/run_experiment.py --seed 42 --output artifacts/run1.json
python src/run_experiment.py --seed 42 --output artifacts/run2.json
python src/run_experiment.py --seed 42 --output artifacts/run3.json
```

The classification metrics and dataset SHA-256 must be identical. Runtime is expected to vary slightly.

## Publication

1. Publish the repository to GitHub/GitLab.
2. Insert the repository URL into `zenodo.json`.
3. Create a Zenodo release and obtain a DOI.
4. Add the DOI badge/link to this README.
