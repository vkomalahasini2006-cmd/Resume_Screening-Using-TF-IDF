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
- **Authentication**: Supabase (Email/Password + Google OAuth)
- **Machine Learning & NLP**:
  - `spaCy` (Skill extraction, Named Entity Recognition)
  - `NLTK` (Text preprocessing, tokenization, lemmatization)
  - `scikit-learn` (TF-IDF Vectorization, Cosine Similarity)
  - `PyPDF2` (PDF Parsing)
  - `pytesseract` (Optional OCR for scanned PDFs)

## 🧠 How the ML System Works

### 1. Resume Text Cleaning & Preprocessing
When a resume (Text or PDF) is uploaded, the system uses **NLTK** to clean the text:
- Converts text to lowercase.
- Removes special characters and punctuation.
- Tokenizes text into individual words.
- Removes standard English stopwords.
- Lemmatizes words to standardize terminology.

### 2. Skill Extraction (NLP)
The job description and candidate resumes are parsed to extract relevant skills using **spaCy**:
- Uses predefined keyword lists for Tech and Soft skills.
- Uses Named Entity Recognition (NER) to find specific tool names.

### 3. Resume-to-Role Similarity Scoring
The core matching engine compares the candidate's resume to the job description:
- Both texts are vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** via `scikit-learn`.
- The **Cosine Similarity** between the job description vector and the resume vector is calculated to find the semantic textual overlap.

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
The FastAPI backend will be available at `http://127.0.0.1:8000`.

### 2. Supabase Authentication Setup
Before running the frontend, you must configure Supabase for Authentication:
1. Create a project on [Supabase](https://supabase.com/).
2. Get your **Project URL** and **Anon Public Key** from Project Settings -> API.
3. Go to **Authentication -> Providers** and enable **Google**. Provide your Google OAuth Client ID and Secret (from Google Cloud Console).
4. Add the following to Google Cloud Console Authorized redirect URIs: `https://<your-project-ref>.supabase.co/auth/v1/callback`
5. In Supabase Authentication -> URL Configuration, set Site URL to `http://localhost:3000` and add `http://localhost:3000/auth/callback` to Redirect URLs.

### 3. Frontend Setup
Navigate to the `frontend` directory, create your environment variables, install dependencies, and start the development server:
```bash
cd frontend

# Create .env.local file
echo "NEXT_PUBLIC_SUPABASE_URL=your_supabase_url" > .env.local
echo "NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key" >> .env.local

npm install
npm run dev
```
The Next.js app will be available at `http://localhost:3000`.

## 🔧 Troubleshooting

### 1. `TypeError: Failed to fetch` on PDF Upload
If you see this error when trying to upload a PDF or screen resumes:
- **Cause 1**: The Python backend is not running. Ensure you ran `python main.py` in the backend folder.
- **Cause 2**: IPv6 resolution issues on Windows. The frontend explicitly calls `http://127.0.0.1:8000` instead of `localhost` to avoid Windows resolving `localhost` to IPv6 (`::1`).
- **Cause 3**: CORS Policies. The backend explicitly allows origins `http://localhost:3000` and `http://127.0.0.1:3000` in `main.py`. Ensure you are accessing the frontend via one of those two exact URLs.

### 2. Error Parsing PDF
- Ensure that the Python backend successfully installed `PyPDF2` (via `pip install -r requirements.txt`). 
- If the PDF is an image/scan rather than text, the backend will attempt to use Tesseract OCR. Ensure `pytesseract` and `pdf2image` are installed if you plan on processing scanned image PDFs.

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
✔ Supabase Authentication integration (Email + Google OAuth)
