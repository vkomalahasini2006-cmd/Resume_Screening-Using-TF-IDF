"""
============================================================
  ML-BASED RESUME SCREENING & RANKING SYSTEM
  Author   : <Your Name>
  Tools    : spaCy · NLTK · scikit-learn · pandas
  Purpose  : Automatically screen, score, and rank resumes
             against a job description using NLP techniques.
============================================================

WORKFLOW
  1. Preprocess text (clean, tokenize, lemmatize)
  2. Extract skills using keyword + spaCy NER
  3. Vectorize with TF-IDF
  4. Score each resume via cosine similarity to JD
  5. Identify skill gaps
  6. Rank candidates and generate a report

USAGE
  pip install spacy nltk scikit-learn pandas tabulate
  python -m spacy download en_core_web_sm
  Then run: python resume_screening_ml.py
"""

# ──────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────
import re
import string
import warnings
warnings.filterwarnings("ignore")

import nltk
import spacy
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (first run only)
for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ──────────────────────────────────────────────
# 1. SAMPLE DATA  (replace with real resumes)
# ──────────────────────────────────────────────

JOB_DESCRIPTION = """
We are looking for a Senior Data Scientist to join our AI team.

Requirements:
- 3+ years of experience in machine learning and data science
- Proficiency in Python, pandas, NumPy, scikit-learn
- Experience with deep learning frameworks: TensorFlow or PyTorch
- Strong skills in SQL and data visualization (Matplotlib, Seaborn, Tableau)
- Experience with NLP techniques and transformer models (BERT, GPT)
- Familiarity with cloud platforms: AWS, GCP, or Azure
- Knowledge of MLOps practices: Docker, Kubernetes, MLflow
- Excellent communication and teamwork skills
- Bachelor's or Master's degree in Computer Science, Statistics, or related field

Responsibilities:
- Build and deploy machine learning models at scale
- Conduct exploratory data analysis and feature engineering
- Collaborate with engineering and product teams
- Present insights to non-technical stakeholders
"""

RESUMES = {
    "Alice Johnson": """
        Senior Data Scientist with 5 years of experience. Expert in Python, pandas,
        NumPy, scikit-learn, TensorFlow, and PyTorch. Built NLP pipelines using BERT
        and transformer models. Proficient in SQL, Tableau, and Matplotlib. Experience
        deploying models on AWS and GCP with Docker and Kubernetes. Published 3 research
        papers. MS in Computer Science. Strong communication and leadership skills.
        Used MLflow for model tracking and experiment management.
    """,
    "Bob Smith": """
        Data Analyst with 2 years of experience. Skilled in Python, pandas, and basic
        machine learning with scikit-learn. Familiar with SQL and Excel. Created
        dashboards in Tableau. Some experience with Matplotlib for data visualization.
        Learning TensorFlow on the side. Bachelor's degree in Statistics.
        Good team player with strong communication skills.
    """,
    "Carol Lee": """
        Machine Learning Engineer with 4 years of experience. Deep expertise in
        PyTorch, TensorFlow, and scikit-learn. Developed production NLP systems using
        BERT and GPT models. Strong Python and SQL skills. Deployed models on Azure
        and AWS using Docker. Experience with MLflow and model monitoring.
        MS in Artificial Intelligence. Published work on transformer architectures.
        Comfortable presenting to technical and non-technical audiences.
    """,
    "David Park": """
        Full Stack Developer with 3 years of experience. Skilled in JavaScript,
        React, Node.js, and PostgreSQL. Some experience with Python scripting.
        Familiar with Docker and Kubernetes for deployment. Used AWS for cloud hosting.
        Bachelor's degree in Computer Engineering. Good problem-solving skills.
        No formal machine learning experience.
    """,
    "Eva Martinez": """
        Data Scientist with 3 years of experience. Proficient in Python, NumPy,
        pandas, scikit-learn, and Seaborn. Built classification and regression models.
        Experience with NLP and text classification tasks. SQL for data extraction.
        Some exposure to TensorFlow. Used GCP for cloud-based data processing.
        Bachelor's degree in Mathematics. Eager to learn deep learning frameworks.
        Strong analytical and communication skills.
    """,
}


# ──────────────────────────────────────────────
# 2. SKILL LIBRARY  (extend this list freely)
# ──────────────────────────────────────────────

TECH_SKILLS = [
    # Programming
    "python", "r", "sql", "java", "scala", "c++", "javascript",
    # ML / DL Frameworks
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    # Data Tools
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "excel",
    # NLP
    "nltk", "spacy", "bert", "gpt", "transformers", "nlp", "hugging face",
    # Cloud
    "aws", "gcp", "azure", "s3", "bigquery", "databricks",
    # MLOps & DevOps
    "docker", "kubernetes", "mlflow", "airflow", "git", "ci/cd",
    # Databases
    "mysql", "postgresql", "mongodb", "redis",
    # Concepts
    "machine learning", "deep learning", "neural networks", "computer vision",
    "natural language processing", "data analysis", "feature engineering",
    "model deployment", "a/b testing", "statistics", "data visualization",
]

SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "collaboration", "problem-solving",
    "analytical", "presentation", "management",
]


# ──────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw text:
      - Lowercase
      - Remove punctuation / special characters
      - Tokenize
      - Remove stopwords
      - Lemmatize
    Returns a single cleaned string.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s+/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    # Keep domain-relevant stop-ish words
    keep = {"not", "no", "nor", "own", "same", "very", "with", "between"}
    stop_words -= keep

    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    return " ".join(tokens)


# ──────────────────────────────────────────────
# 4. SKILL EXTRACTION
# ──────────────────────────────────────────────

def extract_skills(text: str) -> dict:
    """
    Extract tech and soft skills from raw text.
    Uses keyword matching + spaCy NER (ORG entities often catch tool names).
    Returns {'tech': [...], 'soft': [...], 'ner_entities': [...]}.
    """
    lower = text.lower()

    tech_found = sorted({s for s in TECH_SKILLS if s in lower})
    soft_found = sorted({s for s in SOFT_SKILLS if s in lower})

    doc = nlp(text[:1_000_000])  # spaCy has a char limit per call
    ner_entities = list({
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "PERSON")
    })

    return {
        "tech": tech_found,
        "soft": soft_found,
        "ner_entities": ner_entities,
    }


# ──────────────────────────────────────────────
# 5. TFIDF VECTORIZATION & COSINE SIMILARITY
# ──────────────────────────────────────────────

def compute_similarity_scores(jd_text: str, resume_texts: dict) -> dict:
    """
    Vectorize JD + all resumes with TF-IDF.
    Return cosine similarity of each resume to the JD (0–1 scale).
    """
    # Preprocess all texts
    cleaned_jd = preprocess_text(jd_text)
    cleaned_resumes = {k: preprocess_text(v) for k, v in resume_texts.items()}

    corpus = [cleaned_jd] + list(cleaned_resumes.values())

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    return dict(zip(cleaned_resumes.keys(), scores))


# ──────────────────────────────────────────────
# 6. SKILL-OVERLAP BONUS SCORING
# ──────────────────────────────────────────────

def skill_overlap_score(jd_skills: list, candidate_skills: list) -> float:
    """
    Fraction of JD skills present in candidate's resume.
    Acts as an interpretable bonus on top of TF-IDF similarity.
    """
    if not jd_skills:
        return 0.0
    matched = set(jd_skills) & set(candidate_skills)
    return len(matched) / len(jd_skills)


# ──────────────────────────────────────────────
# 7. COMPOSITE SCORER
# ──────────────────────────────────────────────

WEIGHTS = {
    "tfidf_similarity": 0.55,
    "tech_skill_overlap": 0.35,
    "soft_skill_overlap": 0.10,
}

def compute_composite_score(tfidf_sim: float, tech_overlap: float, soft_overlap: float) -> float:
    """Weighted combination of the three scoring signals."""
    score = (
        WEIGHTS["tfidf_similarity"] * tfidf_sim
        + WEIGHTS["tech_skill_overlap"] * tech_overlap
        + WEIGHTS["soft_skill_overlap"] * soft_overlap
    )
    return round(score * 100, 2)  # scale to 0–100


# ──────────────────────────────────────────────
# 8. SKILL GAP ANALYSIS
# ──────────────────────────────────────────────

def identify_skill_gaps(jd_skills: list, candidate_skills: list) -> dict:
    """
    Compare JD-required skills to candidate skills.
    Returns matched, missing, and extra skills.
    """
    jd_set = set(jd_skills)
    cand_set = set(candidate_skills)
    return {
        "matched": sorted(jd_set & cand_set),
        "missing": sorted(jd_set - cand_set),
        "extra":   sorted(cand_set - jd_set),
    }


# ──────────────────────────────────────────────
# 9. MAIN PIPELINE
# ──────────────────────────────────────────────

def screen_resumes(jd: str, resumes: dict) -> pd.DataFrame:
    """
    Full screening pipeline.
    Returns a ranked DataFrame with scores and skill gaps.
    """
    print("\n" + "="*60)
    print("   ML RESUME SCREENING SYSTEM")
    print("="*60)

    # --- TF-IDF similarity ---
    print("\n[1/4] Computing TF-IDF cosine similarity...")
    tfidf_scores = compute_similarity_scores(jd, resumes)

    # --- Skill extraction from JD ---
    print("[2/4] Extracting skills from job description...")
    jd_skills_raw = extract_skills(jd)
    jd_tech_skills = jd_skills_raw["tech"]
    jd_soft_skills = jd_skills_raw["soft"]

    print(f"      JD tech skills found: {jd_tech_skills}")
    print(f"      JD soft skills found: {jd_soft_skills}")

    # --- Score each candidate ---
    print("[3/4] Scoring and ranking candidates...")
    results = []

    for name, resume_text in resumes.items():
        cand_skills = extract_skills(resume_text)
        cand_tech   = cand_skills["tech"]
        cand_soft   = cand_skills["soft"]

        tech_overlap = skill_overlap_score(jd_tech_skills, cand_tech)
        soft_overlap = skill_overlap_score(jd_soft_skills, cand_soft)

        tfidf_sim  = tfidf_scores[name]
        composite  = compute_composite_score(tfidf_sim, tech_overlap, soft_overlap)

        gaps = identify_skill_gaps(jd_tech_skills, cand_tech)

        results.append({
            "Candidate":        name,
            "Composite Score":  composite,
            "TF-IDF Sim (%)":   round(tfidf_sim * 100, 1),
            "Tech Overlap (%)": round(tech_overlap * 100, 1),
            "Soft Overlap (%)": round(soft_overlap * 100, 1),
            "Tech Skills":      ", ".join(cand_tech) if cand_tech else "—",
            "Matched Skills":   ", ".join(gaps["matched"]) if gaps["matched"] else "—",
            "Missing Skills":   ", ".join(gaps["missing"]) if gaps["missing"] else "None ✓",
        })

    # --- Rank ---
    df = (
        pd.DataFrame(results)
        .sort_values("Composite Score", ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1  # 1-based rank
    df.index.name = "Rank"

    print("[4/4] Done.\n")
    return df


# ──────────────────────────────────────────────
# 10. REPORTING
# ──────────────────────────────────────────────

def tier_label(score: float) -> str:
    if score >= 70:  return "★ STRONG FIT"
    if score >= 50:  return "◑ GOOD FIT"
    if score >= 30:  return "△ PARTIAL FIT"
    return                  "✗ WEAK FIT"

def print_report(df: pd.DataFrame) -> None:
    """Pretty-print a screening report to stdout."""
    print("="*60)
    print("   CANDIDATE RANKING REPORT")
    print("="*60)

    for rank, row in df.iterrows():
        tier = tier_label(row["Composite Score"])
        print(f"\n  #{rank}  {row['Candidate']}  [{tier}]")
        print(f"      Score        : {row['Composite Score']:.1f} / 100")
        print(f"      TF-IDF Sim   : {row['TF-IDF Sim (%)']:.1f}%")
        print(f"      Tech Overlap : {row['Tech Overlap (%)']:.1f}%")
        print(f"      Soft Overlap : {row['Soft Overlap (%)']:.1f}%")
        print(f"      Matched      : {row['Matched Skills']}")
        print(f"      Missing      : {row['Missing Skills']}")

    print("\n" + "="*60)
    print("  TOP RECOMMENDATION:", df.iloc[0]["Candidate"])
    print("="*60 + "\n")


# ──────────────────────────────────────────────
# 11. OPTIONAL: VISUALIZATION
# ──────────────────────────────────────────────

def visualize_results(df: pd.DataFrame) -> None:
    """Bar chart of composite scores (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors = [
            "#16a34a" if s >= 70 else
            "#2563eb" if s >= 50 else
            "#d97706" if s >= 30 else
            "#dc2626"
            for s in df["Composite Score"]
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(df["Candidate"], df["Composite Score"], color=colors, height=0.5)

        ax.set_xlabel("Composite Score (0–100)", fontsize=12)
        ax.set_title("Resume Screening Results – Candidate Ranking", fontsize=14, fontweight="bold")
        ax.axvline(70, color="#16a34a", linestyle="--", linewidth=1, alpha=0.6, label="Strong threshold (70)")
        ax.axvline(50, color="#2563eb", linestyle=":",  linewidth=1, alpha=0.6, label="Good threshold (50)")
        ax.invert_yaxis()
        ax.set_xlim(0, 100)

        for bar, score in zip(bars, df["Composite Score"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{score:.1f}", va="center", fontsize=11)

        legend_patches = [
            mpatches.Patch(color="#16a34a", label="Strong Fit (≥70)"),
            mpatches.Patch(color="#2563eb", label="Good Fit (≥50)"),
            mpatches.Patch(color="#d97706", label="Partial Fit (≥30)"),
            mpatches.Patch(color="#dc2626", label="Weak Fit (<30)"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig("resume_screening_results.png", dpi=150)
        plt.show()
        print("Chart saved to resume_screening_results.png")
    except ImportError:
        print("matplotlib not installed – skipping visualization.")


# ──────────────────────────────────────────────
# 12. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ranked_df = screen_resumes(JOB_DESCRIPTION, RESUMES)
    print_report(ranked_df)

    # Save CSV
    ranked_df.to_csv("ranked_candidates.csv")
    print("Full results saved to ranked_candidates.csv")

    # Optional chart
    visualize_results(ranked_df)
