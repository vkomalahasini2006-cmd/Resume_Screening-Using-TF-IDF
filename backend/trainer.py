"""Train a multi-class resume classifier from a Kaggle CSV.
Saves TF-IDF vectorizer, LabelEncoder, and classifier to `backend/models/`.
"""
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from backend.dataset_loader import load_kaggle_resume_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
LBL_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
KEYWORDS_PATH = os.path.join(MODEL_DIR, "class_keywords.joblib")


def train_from_csv(csv_path: str, max_samples: int = 2000) -> Tuple[str, int]:
    """Train and persist the classifier. Returns (model_dir, n_samples)."""
    resumes, df = load_kaggle_resume_dataset(csv_path, max_samples)
    texts = list(resumes.values())
    labels = []
    # Try to get category column from df
    cat_col = None
    for c in ["Category", "category", "Label", "label"]:
        if c in df.columns:
            cat_col = c
            break
    if cat_col is None:
        raise ValueError("CSV missing category column (Category)")

    labels = df[cat_col].astype(str).head(len(texts)).tolist()

    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Ensure model dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save artifacts
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(clf, CLF_PATH)
    joblib.dump(le, LBL_PATH)

    # Compute top keywords per class
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_keywords = {}
    coefs = clf.coef_  # shape (n_classes, n_features)
    for class_idx, class_label in enumerate(le.classes_):
        top_idx = np.argsort(coefs[class_idx])[::-1][:50]
        keywords = feature_names[top_idx].tolist()
        class_keywords[class_label] = keywords

    joblib.dump(class_keywords, KEYWORDS_PATH)

    return MODEL_DIR, len(texts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to Resume.csv')
    parser.add_argument('--limit', type=int, default=2000)
    args = parser.parse_args()
    model_dir, n = train_from_csv(args.csv, max_samples=args.limit)
    print(f"Trained classifier on {n} samples and saved artifacts to {model_dir}")
