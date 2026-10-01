#!/usr/bin/env python3
"""Score a predicted-labels CSV with the Silhouette Score.

Usage:
    python3 scoring.py [pred_label.csv]

pred_label.csv must have the same format as sample_submission.csv: an "id"
column and a "label" column, one row per pal id, with ids exactly
0..n_samples-1 so they align with features.npy. The score is fully
unsupervised (no ground-truth labels are needed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

DATA_DIR = Path(__file__).resolve().parent


def load_features(path=None):
    """Returns (n_samples, 60) float32 array; row i == pal with id i."""
    path = path or DATA_DIR / 'features.npy'
    return np.load(path)


def score_submission(pred_path, label_col='label'):
    """Score a predictions CSV (id,label) with the Silhouette Score against
    features.npy. Returns {'silhouette', 'n_samples'}."""
    pred = pd.read_csv(pred_path)
    features = load_features()

    # features.npy row i is pal id i, so sort predictions by id to align with it
    pred_sorted = pred.sort_values('id')
    if not np.array_equal(pred_sorted['id'].to_numpy(), np.arange(features.shape[0])):
        raise ValueError(f'{pred_path} ids must be exactly 0..{features.shape[0] - 1} to align with features.npy')

    y_pred = pred_sorted[label_col].to_numpy()
    if len(set(y_pred)) > 1:
        silhouette = silhouette_score(features, y_pred)
    else:
        silhouette = float('nan')  # undefined with a single cluster

    return {
        'silhouette': silhouette,
        'n_samples': len(pred_sorted),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path', nargs='?', default=str(DATA_DIR / 'pred_label.csv'),
                         help='Predictions CSV to score (id,label). Default: pred_label.csv')
    args = parser.parse_args()

    scores = score_submission(args.pred_path)

    print(f"n_samples={scores['n_samples']}")
    print(f"Silhouette Score:     {scores['silhouette']:.4f}")
