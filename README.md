# ResumeIQ Pro - ML Resume Screening System

An advanced Machine Learning-based resume screening system that automatically reads, scores, and ranks candidate resumes against a given job description. 

## 🎯 Objective
Hiring teams receive hundreds of resumes for a single job role. Manually reading each resume is slow, inconsistent, and error-prone. ResumeIQ Pro uses Natural Language Processing (NLP) to automate this process, allowing recruiters to:
- Shortlist candidates faster
- Match skills with job requirements
- Identify missing or weak skills
- Reduce workload and avoid human bias

## 🛠️ Tech Stack
- **Frontend**: Next.js, React (Black and Yellow Premium Theme)
- **Backend**: FastAPI, Python
- **Database**: MongoDB (Motor Async)
- **Machine Learning & NLP**:
  - `spaCy` (Skill extraction, Named Entity Recognition)
  - `NLTK` (Text preprocessing, tokenization, lemmatization)
  - `scikit-learn` (TF-IDF Vectorization, Cosine Similarity)
  - `PyPDF2` (PDF Parsing)

## 🧠 How the ML System Works

### 1. Resume Text Cleaning & Preprocessing
When a resume (Text or PDF) is uploaded, the system uses **NLTK** to clean the text:
- Converts text to lowercase.
- Removes special characters and punctuation.
- Tokenizes text into individual words.
- Removes standard English stopwords (except domain-relevant words like "not" or "with").
- Lemmatizes words (e.g., converts "running" to "run") to standardize terminology.

### 2. Skill Extraction (NLP)
The job description and candidate resumes are parsed to extract relevant skills using **spaCy**:
- Uses predefined keyword lists for Tech and Soft skills.
- Uses Named Entity Recognition (NER) to find specific tool names or organizations labeled as `ORG` or `PRODUCT`.

### 3. Resume-to-Role Similarity Scoring
The core matching engine compares the candidate's resume to the job description:
- Both texts are vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** via `scikit-learn`. This evaluates how important a word is to a document in a collection.
- The **Cosine Similarity** between the job description vector and the resume vector is calculated to find the semantic textual overlap (scored from 0 to 1).

### 4. Skill Gap Identification
The system cross-references the skills extracted from the job description with the skills found in the candidate's resume. It produces:
- **Matched Skills**: Skills present in both the JD and the Resume.
- **Missing Skills**: Crucial skills required by the JD but absent from the candidate's profile.

### 5. Candidate Ranking Based on Role Fit
Each candidate is assigned a **Composite Score (0-100)** which acts as their final ranking metric. The score is a weighted combination of:
- **TF-IDF Textual Similarity** (55% weight)
- **Technical Skill Overlap** (35% weight)
- **Soft Skill Overlap** (10% weight)

*Candidates with a score ≥ 70 are labeled as a "Strong Fit".*

## 🚀 How to Run Locally

### 1. Backend Setup
Navigate to the `backend` directory, install dependencies, and run the server:
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```
The FastAPI backend will be available at `http://localhost:8000`.

If you prefer to use Uvicorn directly:
```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend Setup
Navigate to the `frontend` directory, install dependencies, and start the development server:
```bash
cd frontend
npm install
npm run dev
```
The Next.js app will be available at `http://localhost:3000`.

### 3. Dataset-driven Resume Verification
This project uses the Kaggle resume dataset to derive class-specific keywords for each predicted job class.
The backend route `POST /api/verify-resume` accepts:
```json
{
  "jd": "<job description text>",
  "resume_text": "<resume text or extracted PDF text>"
}
```
It returns:
- `prediction`: predicted resume/job class
- `dataset_class_keywords`: dataset-derived keywords for that class
- `resume_matched_keywords`: keywords found in the resume
- `resume_missing_keywords`: dataset keywords missing from the resume
- `resume_coverage_percent`: resume coverage of dataset keywords

### 4. Using the Kaggle Resume Dataset
Download `Resume.csv` from the Kaggle dataset at `https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset`.
Then run the screening script from the repo root:
```bash
python resume_screening_ml.py --dataset backend/data/Resume.csv --limit 100
```
If you store the CSV elsewhere, pass its full path instead of `backend/data/Resume.csv`.

The script will produce `ranked_candidates.csv` in the current directory.

## 📤 Deliverables Met
✔ Resume text cleaning & preprocessing
✔ Skill extraction using NLP
✔ Job description parsing
✔ Resume-to-role similarity scoring
✔ Candidate ranking based on role fit
✔ Skill gap identification
✔ Visual comparisons & detailed candidate summaries
