import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# We default to local MongoDB if MONGODB_URL is not set
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URL, tlsCAFile=certifi.where() if "mongodb+srv" in MONGODB_URL else None)
db = client["resume_iq"]
screenings_collection = db["screenings"]
users_collection = db["users"]

# ─────────────────────────────────────────────
# SCREENING OPERATIONS
# ─────────────────────────────────────────────

async def save_screening(jd_text: str, jd_skills: list, candidates: list, top_recommendation: str, analysis_note: str):
    """Save a screening session to the database."""
    document = {
        "timestamp": datetime.utcnow(),
        "job_description": jd_text,
        "jd_skills": jd_skills,
        "candidates": candidates,
        "top_recommendation": top_recommendation,
        "analysis_note": analysis_note
    }
    result = await screenings_collection.insert_one(document)
    return str(result.inserted_id)

async def get_history(limit: int = 20):
    """Retrieve past screening sessions."""
    cursor = screenings_collection.find(
        {},
        {"_id": 1, "timestamp": 1, "top_recommendation": 1, "candidates": 1, "job_description": 1, "jd_skills": 1, "analysis_note": 1}
    ).sort("timestamp", -1).limit(limit)
    history = []
    async for document in cursor:
        document["_id"] = str(document["_id"])
        if "timestamp" in document and isinstance(document["timestamp"], datetime):
            document["date"] = document["timestamp"].isoformat()
        else:
            document["date"] = "Unknown"
        history.append(document)
    return history

# ─────────────────────────────────────────────
# USER / AUTH OPERATIONS
# ─────────────────────────────────────────────

async def create_user(name: str, email: str, hashed_password: str):
    """Insert a new user into the database."""
    document = {
        "name": name,
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
    }
    result = await users_collection.insert_one(document)
    return str(result.inserted_id)

async def get_user_by_email(email: str):
    """Find a user by email address."""
    user = await users_collection.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
    return user

async def email_exists(email: str) -> bool:
    """Check if an email is already registered."""
    doc = await users_collection.find_one({"email": email}, {"_id": 1})
    return doc is not None
