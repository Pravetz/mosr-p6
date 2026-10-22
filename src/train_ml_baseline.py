from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/benchmark_specs_pr4.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='artifacts/ml_baseline_metrics.json')
    ap.add_argument('--use-mlflow', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = df['gold_invalid'].astype(int)
    feature_cols = ['task','split','seed','preprocess_fit','tune_on','metric','cv','cv_preprocess',
                    'augment_on','dataset_kind','group_split','time_shuffle']
    X = df[feature_cols].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=args.seed
    )
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)])
    model = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000, random_state=args.seed))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0),
        'seed': args.seed,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    if args.use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
        mlflow.set_experiment('mlspec-ml-baseline')
        with mlflow.start_run():
            mlflow.log_params({'seed': args.seed, 'model': 'LogisticRegression', 'test_size': 0.30})
            for k in ['accuracy','precision','recall','f1']:
                mlflow.log_metric(k, float(metrics[k]))
            mlflow.log_artifact(args.output)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
