import json
from backend.model_api import classify_text

JD = '''Build and productionize machine learning models and ML systems that solve real-world problems across NLP, CV, and tabular data. Own model development, evaluation, deployment, and monitoring.

Key responsibilities:
- Design, implement, and validate ML models (classification, regression, ranking, generation).
- Clean, explore, and transform datasets; build reproducible data pipelines.
- Train models using scikit-learn, PyTorch, or TensorFlow and tune hyperparameters.
- Package models for production: CI/CD, containerization, inference APIs, and monitoring.
- Collaborate with product and engineering to translate requirements into ML solutions.
- Implement A/B tests, model performance tracking, and drift detection.
- Write clear documentation, reproducible experiments, and production-ready code.

Required qualifications:
- 3+ years experience building ML systems or related research experience.
- Strong Python skills and familiarity with libraries: scikit-learn, PyTorch/TensorFlow, pandas.
- Solid understanding of ML fundamentals: feature engineering, model selection, evaluation metrics, and statistical reasoning.
- Experience with ML engineering practices: packaging, containerization (Docker), and deployment.
- Experience working with real-world datasets and data engineering basics (SQL, ETL).

Nice-to-have:
- Experience with cloud platforms (AWS/GCP/Azure), ML infra (MLflow, Kubeflow), or MLOps tooling.
- Background in NLP, Transformers, computer vision, or time-series modeling.
- Experience productionizing inference at scale and implementing monitoring/ alerting.
- Advanced degree (MS/PhD) in CS, ML, statistics, or related field.

How to apply / test:
- Submit resume and a short description of a recent ML project; include code or a reproducible notebook if available.'''

res = classify_text(JD)

# Compute missing dataset-derived keywords
jd_lower = JD.lower()
class_keywords = res.get('class_keywords', [])
missing = [kw for kw in class_keywords if kw.lower() not in jd_lower]

out = {
    'prediction': res.get('prediction'),
    'top_probabilities': dict(sorted(res.get('probabilities', {}).items(), key=lambda x: -x[1])[:5]),
    'top_features': res.get('top_features', [])[:10],
    'recommendations_missing_keywords': missing[:40]
}

print(json.dumps(out, indent=2))
