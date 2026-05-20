from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io

from pdf_parser import parse_pdf_with_ocr
from model_api import classify_text, get_all_class_keywords, verify_resume_against_jd, ModelNotReady

from ml_engine import (
    extract_skills,
    compute_similarity_scores,
    skill_overlap_score,
    compute_composite_score,
    identify_skill_gaps
)

from database import save_screening, get_history

app = FastAPI(title="Resume Screening API")

# Configure CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeInput(BaseModel):
    name: str
    text: str

class ScreenRequest(BaseModel):
    jd: str
    resumes: List[ResumeInput]

@app.post("/api/screen")
async def screen_resumes_api(request: ScreenRequest):
    jd_text = request.jd
    resumes_list = request.resumes
    
    if not jd_text or not resumes_list:
        return {"error": "Job description and resumes are required."}
        
    # Prepare resumes dict for similarity computation
    resume_texts = {r.name: r.text for r in resumes_list}
    
    # 1. Compute TF-IDF similarity
    tfidf_scores = compute_similarity_scores(jd_text, resume_texts)
    
    # 2. Extract JD skills
    jd_skills_raw = extract_skills(jd_text)
    jd_tech_skills = jd_skills_raw["tech"]
    jd_soft_skills = jd_skills_raw["soft"]
    
    candidates_result = []
    
    # 3. Score each candidate
    for r in resumes_list:
        name = r.name
        text = r.text
        
        cand_skills = extract_skills(text)
        cand_tech = cand_skills["tech"]
        cand_soft = cand_skills["soft"]
        
        tech_overlap = skill_overlap_score(jd_tech_skills, cand_tech)
        soft_overlap = skill_overlap_score(jd_soft_skills, cand_soft)
        
        tfidf_sim = tfidf_scores[name]
        composite = compute_composite_score(tfidf_sim, tech_overlap, soft_overlap)
        
        gaps = identify_skill_gaps(jd_tech_skills, cand_tech)
        
        # Simple rule-based logic for strengths/concerns
        strengths = []
        if tech_overlap >= 0.7: strengths.append("Strong technical skill overlap with JD.")
        if soft_overlap >= 0.5: strengths.append("Demonstrates good soft skills.")
        if tfidf_sim >= 0.3: strengths.append("High overall textual relevance to the role.")
        if not strengths: strengths.append("General experience aligns somewhat with role.")
        
        concerns = []
        if tech_overlap < 0.4: concerns.append(f"Missing key technical skills ({len(gaps['missing'])} missing).")
        if tfidf_sim < 0.1: concerns.append("Resume language differs significantly from JD.")
        
        summary = "Strong candidate with solid technical foundation." if composite >= 70 else \
                  "Good candidate, but has some skill gaps." if composite >= 50 else \
                  "Candidate lacks several key requirements."
                  
        candidates_result.append({
            "name": name,
            "composite_score": composite,
            "tfidf_similarity": round(tfidf_sim * 100, 1),
            "tech_skill_overlap": round(tech_overlap * 100, 1),
            "soft_skill_overlap": round(soft_overlap * 100, 1),
            "matched_skills": gaps["matched"],
            "missing_skills": gaps["missing"],
            "extra_skills": gaps["extra"],
            "summary": summary,
            "strengths": strengths,
            "concerns": concerns
        })
        
    # Sort candidates by score descending
    candidates_result.sort(key=lambda x: x["composite_score"], reverse=True)
    
    top_recommendation = candidates_result[0]["name"] if candidates_result else "None"
    
    analysis_note = f"Analysis complete. {len(candidates_result)} candidates screened. Top candidate '{top_recommendation}' scored {candidates_result[0]['composite_score'] if candidates_result else 0}/100."
    
    # Save the screening result to MongoDB asynchronously
    await save_screening(
        jd_text=jd_text,
        jd_skills=jd_tech_skills + jd_soft_skills,
        candidates=candidates_result,
        top_recommendation=top_recommendation,
        analysis_note=analysis_note
    )
    
    return {
        "candidates": candidates_result,
        "jd_key_skills": jd_tech_skills + jd_soft_skills,
        "top_recommendation": top_recommendation,
        "analysis_note": analysis_note
    }

@app.get("/api/history")
async def get_screening_history():
    history = await get_history()
    return {"history": history}

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "File must be a PDF."}
    try:
        content = await file.read()
        result = await parse_pdf_with_ocr(content, file.filename)
        return result
    except Exception as e:
        return {"error": f"Failed to parse PDF: {str(e)}", "debug_info": {"exception": type(e).__name__}}


@app.post("/api/classify")
async def classify_endpoint(payload: dict = None, file: UploadFile = File(None)):
    """Classify provided text or uploaded PDF against trained dataset classifier.
    Accepts JSON `{"text": "..."}` or a PDF file upload.
    """
    text = None
    if payload and isinstance(payload, dict):
        text = payload.get("text")

    if file is not None:
        try:
            content = await file.read()
            parsed = await parse_pdf_with_ocr(content, file.filename)
            text = parsed.get("text", "") if isinstance(parsed, dict) else ""
        except Exception as e:
            return {"error": f"Failed to parse uploaded file: {str(e)}"}

    if not text:
        return {"error": "No text provided for classification."}

    try:
        result = classify_text(text)
        return result
    except ModelNotReady as mr:
        return {"error": str(mr)}
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}


class VerifyResumeRequest(BaseModel):
    jd: str
    resume_text: str


@app.post("/api/verify-resume")
async def verify_resume(request: VerifyResumeRequest):
    if not request.jd or not request.resume_text:
        return {"error": "Job description and resume text are required."}

    try:
        result = verify_resume_against_jd(request.jd, request.resume_text)
        return result
    except ModelNotReady as mr:
        return {"error": str(mr)}
    except Exception as e:
        return {"error": f"Resume verification failed: {str(e)}"}


@app.get("/api/classifier-info")
async def classifier_info():
    try:
        keywords = get_all_class_keywords()
        return {"class_keywords": keywords}
    except ModelNotReady as mr:
        return {"error": str(mr)}
    except Exception as e:
        return {"error": f"Failed to load classifier info: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
