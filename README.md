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

Start the local tracking server (keep this terminal open):

```bash
make mlflow
```

In another terminal, log either experiment to that server:

```bash
make mlflow-run
make mlflow-baseline
```

Open `http://127.0.0.1:5000`. Local MLflow state is stored under the ignored
`.mlflow/` directory. Run `make clean` as your normal user; `sudo` is not needed.

## Docker

```bash
docker build -t mlspec-experiment:v1 .
docker run --rm mlspec-experiment:v1
```

The standalone image runs the evaluation without MLflow tracking.

Or run the experiment and the MLflow server together:

```bash
make up
```

Docker stores its MLflow database and artifacts in the `mlflow-data` named
volume instead of writing container-owned files into the repository. Stop the
containers with `make down`. To also delete the Docker MLflow history, run
`make clean-docker`.

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
