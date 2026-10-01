#!/usr/bin/env python3
"""Basic data loading utilities for the HW1 pal dataset.

Every pal is addressed by a numeric id (0..286), assigned in alphabetical
order by pal_id. Row i of features.npy corresponds to id i.

Files in this folder:
  features.npy            - (n_samples, 60) float32 feature matrix, row i = id i
                             (color feature + CLIP image feature, standardized)
                             This is the only file the default pipeline uses.
  sample_submission.csv    - id,label template (written by clustering,
                             called from __main__ below) showing the exact format
                             a Kaggle submission should have

  Extra files (optional, for bonus exploration -- not used by default):
  pal_images/<id>.png     - artwork, one file per pal, named by numeric id
  metadata.json            - list of per-pal numeric attributes
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent


def load_features(path=None):
    """Returns (n_samples, 60) float32 array; row i == pal with id i."""
    path = path or DATA_DIR / 'features.npy'
    return np.load(path)


def load_metadata(path=None):
    """Bonus: returns a list of dicts, one per pal, ordered by id (sorted ascending)."""
    path = path or DATA_DIR / 'metadata.json'
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    return sorted(rows, key=lambda r: r['id'])


def load_image(pal_id, images_dir=None):
    """Bonus: pal_id here is the numeric id (int)."""
    images_dir = images_dir or DATA_DIR / 'pal_images'
    return Image.open(Path(images_dir) / f'{pal_id}.png')


def load_all():
    """Load features, indexed by id (0..n_samples-1)."""
    features = load_features()
    ids = list(range(features.shape[0]))
    return ids, features


def clustering(path=None, label_col='label'):
    """Write a Kaggle-style sample_submission.csv: id + a predicted cluster
    label for every id, so the file has the exact columns and row count a
    real submission must have.

    TODO: replace the placeholder below with your own clustering algorithm
    (e.g. k-means) over `features`. Currently every id is assigned to the
    same cluster (0), which is just a valid-format placeholder.
    """
    path = path or DATA_DIR / 'sample_submission.csv'
    ids = list(range(load_features().shape[0]))
    labels = [0] * len(ids)  

    # ===================================================
    # TODO: replace with your predicted cluster labels
    # ===================================================


    df = pd.DataFrame({'id': ids, label_col: labels})
    df.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    ids, features = load_all()
    print(f'{len(ids)} pals, feature dim = {features.shape[1]}')

    out_path = clustering()
    print(f'wrote {out_path}')
