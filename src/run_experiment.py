from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import pandas as pd
from provenance import capture_git_provenance
from verifier import evaluate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/benchmark_specs.csv')
    ap.add_argument('--output', default='artifacts/metrics.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use-mlflow', action='store_true')
    args = ap.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    t0 = time.perf_counter()
    metrics = evaluate(df)
    elapsed = time.perf_counter() - t0
    metrics.update({
        'seed': args.seed,
        'elapsed_s': elapsed,
        'data_sha256': sha256(data_path),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        **capture_git_provenance(),
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.use_mlflow:
        import mlflow
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('mlspec-methodological-validity')
        with mlflow.start_run():
            mlflow.log_params({
                'seed': args.seed,
                'n_specs': metrics['n'],
                'validator': 'full',
                'data_sha256': metrics['data_sha256'],
                'git_commit': metrics['git_commit'],
                'git_dirty': metrics['git_dirty'],
            })
            for key in ['accuracy','precision','recall','f1','elapsed_s']:
                mlflow.log_metric(key, float(metrics[key]))
            mlflow.log_artifact(str(out))
            mlflow.log_artifact(str(data_path), artifact_path='data_snapshot')

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
