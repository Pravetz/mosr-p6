# MLSpec methodological-validity reproducible artifact

This repository accompanies the study **"Method of formal specification and preventive
verification of methodological validity of machine-learning experiments"** and the
Practical Work 5/6 reproducibility exercises.

It contains two related but distinct workflows:

1. A reproducible controlled experiment that parses a small line-oriented DSL and checks
   eight methodological-validity rules.
2. A corrected reanalysis of the preserved Practical Work 5 measurements.

The controlled benchmark demonstrates rule coverage. It is not evidence that the verifier
generalizes to arbitrary real-world ML projects.

## Repository contents

- `src/generate_benchmark.py` - deterministic generator for 64 paired valid/invalid base scenarios.
- `data/benchmark_specs.csv` - 128 unique generated specifications used by default.
- `src/verifier.py` - DSL parser plus eight semantic validation rules.
- `src/run_experiment.py` - deterministic rule-based evaluation and optional MLflow logging.
- `src/train_ml_baseline.py` - group-separated Logistic Regression baseline with no base-scenario overlap.
- `notebooks/pr5_statistical_analysis.ipynb` - corrected version of the earlier notebook.
- `notebooks/archive/pr5_original.ipynb` - unchanged supplied notebook retained for provenance.
- `data/pr5/` - preserved PR5 CSV inputs and historical derived tables.
- `src/analyze_pr5.py` - scripted corrected PR5 analysis.
- `Dockerfile` and `docker-compose.yml` - isolated experiment and MLflow service.
- `data/benchmark_specs_pr4.csv.dvc` - DVC pointer for the historical PR4 benchmark.
- `.zenodo.json` - GitHub/Zenodo release metadata.
- `tests/` - parser, benchmark-integrity and statistical-analysis regression tests.

## Clone and install

```bash
git clone https://github.com/Pravetz/mosr-p6.git
cd mosr-p6
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the PR5 notebook and statistical analysis:

```bash
pip install -r requirements-analysis.txt
```

For the test suite:

```bash
pip install -r requirements-dev.txt
```

## Reproduce the controlled experiment

```bash
make data
make run
make baseline
```

Expected properties with seed 42:

- the generated benchmark contains 128 unique rows, 64 valid and 64 invalid;
- the rule verifier reports TP=64, TN=64, FP=0 and FN=0;
- the ML split contains 44 training and 20 test base scenarios with `group_overlap=0`;
- each JSON result includes the dataset SHA-256 and Git provenance.

The ML baseline can still score perfectly because the synthetic violations are explicit
categorical rule signals. The group-separated split establishes that this is not caused by
duplicated test rows, but it does not establish external validity.

## Reproduce the Practical Work 5 analysis

```bash
make analyze
make notebook                    # optional interactive notebook
```

Corrected outputs are written under `artifacts/pr5/`. The ten stored rows per
dataset/algorithm have identical F1 and accuracy values, so the corrected analysis collapses
them to five independent dataset blocks before Friedman and Nemenyi testing. The original
CSV measurements are preserved unchanged. Their upstream measurement-generation program was
not present in the supplied notebook/archive, so this repository reproduces their analysis,
not their original acquisition.

## MLflow locally

Keep the server running in one terminal:

```bash
make mlflow
```

In another activated environment:

```bash
make mlflow-run
make mlflow-baseline
```

Open <http://127.0.0.1:5000>. The baseline run logs the fitted scikit-learn pipeline as a
model artifact; both runs log parameters, metrics, dataset hashes and Git provenance.

## Docker Compose

```bash
make up
```

If your account cannot access `/var/run/docker.sock`, elevate Docker only (do not run all of
Make as root):

```bash
make up DOCKER='sudo docker'
```

Compose builds the image, waits for the MLflow health check, runs the experiment, and stores
MLflow state in the `mlflow-data` named volume rather than creating root-owned repository
files. Open <http://127.0.0.1:5000> while the service is running.

```bash
make down DOCKER='sudo docker'          # keep MLflow history
make clean-docker DOCKER='sudo docker'  # delete containers and the MLflow named volume
```

A standalone non-MLflow evaluation is also available:

```bash
docker build --build-arg GIT_COMMIT="$(git rev-parse HEAD)" -t mlspec-experiment:v1 .
docker run --rm mlspec-experiment:v1
```

## Tests

```bash
make test
```

Tests verify deterministic benchmark generation, absence of duplicate feature vectors,
balanced paired groups, DSL parsing, all eight validation rules, and the corrected PR5
experimental unit.

## Reproducibility check

```bash
make reproduce
```

Metrics and dataset hashes must be identical. Runtime is expected to vary. For release
artifacts, run from a clean tagged checkout and confirm `git_dirty` is `false`.
