from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Violation:
    code: str
    message: str


def validate_row(row: pd.Series) -> List[Violation]:
    v: List[Violation] = []
    # Local consistency / task-metric rules
    if row['task'] == 'classification' and row['metric'] in {'mse','rmse','mae','r2'}:
        v.append(Violation('E201', 'Regression metric used for classification task'))
    if row['task'] == 'regression' and row['metric'] in {'accuracy','precision','recall','f1','roc_auc'}:
        v.append(Violation('E202', 'Classification metric used for regression task'))

    # Provenance / leakage rules
    if row['preprocess_fit'] != 'train':
        v.append(Violation('E101', 'Preprocessor is fitted outside the training partition'))
    if row['tune_on'] == 'test':
        v.append(Violation('E102', 'Test partition is used for model selection or hyperparameter tuning'))
    if row['cv'] != 'none' and row['cv_preprocess'] != 'inside_fold':
        v.append(Violation('E103', 'Cross-validation preprocessing is fitted outside each fold'))
    if row['augment_on'] != 'train':
        v.append(Violation('E104', 'Augmentation is applied outside the training partition'))

    # Partition semantics
    if row['dataset_kind'] == 'grouped' and row['group_split'] != 'required':
        v.append(Violation('E301', 'Grouped observations may cross train/test partitions'))
    if row['dataset_kind'] == 'temporal' and row['time_shuffle'] != 'false':
        v.append(Violation('E302', 'Temporal order is violated by shuffling'))

    # Reproducibility
    seed = str(row['seed']).strip().lower()
    if seed in {'', 'none', 'nan', 'null'}:
        v.append(Violation('W401', 'Random seed is not fixed'))
    return v


def evaluate(df: pd.DataFrame) -> dict:
    y_true = df['gold_invalid'].astype(bool).to_numpy()
    y_pred = df.apply(lambda r: len(validate_row(r)) > 0, axis=1).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(df)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'n': int(len(df)),
    }
