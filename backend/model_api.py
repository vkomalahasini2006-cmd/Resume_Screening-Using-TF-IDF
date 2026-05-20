"""Model load / inference helpers for the resume classifier."""
import os
import re
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
LBL_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
KEYWORDS_PATH = os.path.join(MODEL_DIR, "class_keywords.joblib")


class ModelNotReady(Exception):
    pass


def load_artifacts() -> Tuple[TfidfVectorizer, Any, Any, Dict[str, List[str]]]:
    if not (os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH) and os.path.exists(LBL_PATH)):
        raise ModelNotReady("Model artifacts not found. Train the classifier first using backend/trainer.py")

    vectorizer = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    le = joblib.load(LBL_PATH)
    class_keywords = joblib.load(KEYWORDS_PATH) if os.path.exists(KEYWORDS_PATH) else {}

    return vectorizer, clf, le, class_keywords


def classify_text(text: str, top_k_features: int = 10) -> Dict[str, Any]:
    vec, clf, le, class_keywords = load_artifacts()
    X = vec.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    prob_scores = dict(zip(le.classes_.tolist(), probs.tolist()))

    # feature contribution explanation for predicted class
    try:
        feature_names = vec.get_feature_names_out()
        coef = clf.coef_[pred_idx]
        x = X.toarray()[0]
        contrib = coef * x
        top_idx = np.argsort(contrib)[::-1][:top_k_features]
        top_features = [(feature_names[i], float(contrib[i])) for i in top_idx if contrib[i] > 0]
    except Exception:
        top_features = []

    return {
        "prediction": pred_label,
        "probabilities": prob_scores,
        "top_features": top_features,
        "class_keywords": class_keywords.get(pred_label, [])
    }


def _normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower()


def _keyword_in_text(text: str, keyword: str) -> bool:
    keyword = keyword.strip()
    if not keyword:
        return False
    pattern = re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")
    return bool(pattern.search(text.lower()))


def verify_resume_against_jd(jd_text: str, resume_text: str, top_k_keywords: int = 50) -> Dict[str, Any]:
    """Classify JD, derive dataset keywords for that class, and verify resume coverage."""
    classification = classify_text(jd_text, top_k_features=top_k_keywords)
    class_keywords = classification.get("class_keywords", [])
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    jd_missing = [kw for kw in class_keywords if not _keyword_in_text(jd_lower, kw)]
    resume_matched = [kw for kw in class_keywords if _keyword_in_text(resume_lower, kw)]
    resume_missing = [kw for kw in class_keywords if kw not in resume_matched]

    resume_score = round((len(resume_matched) / max(len(class_keywords), 1)) * 100, 2)

    return {
        "prediction": classification["prediction"],
        "probabilities": classification["probabilities"],
        "top_features": classification["top_features"],
        "dataset_class_keywords": class_keywords,
        "jd_missing_keywords": jd_missing,
        "resume_matched_keywords": resume_matched,
        "resume_missing_keywords": resume_missing,
        "resume_coverage_percent": resume_score,
    }


def get_all_class_keywords() -> Dict[str, List[str]]:
    _, _, _, class_keywords = load_artifacts()
    return class_keywords
