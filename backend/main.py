from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "MyStudyMate v2 backend is live!"}

class TOCRequest(BaseModel):
    transcript: str

class TOCEntry(BaseModel):
    heading: str
    subtopics: List[str]

@app.post("/generate-toc", response_model=List[TOCEntry])
def generate_toc(data: TOCRequest):
    # 🔧 Fake/mock TOC entries (replace with GPT later)
    return [
        {
            "heading": "1. Introduction to AI",
            "subtopics": ["What is AI?", "Brief history of AI"]
        },
        {
            "heading": "2. Supervised Learning",
            "subtopics": ["Labeled data", "Classification vs Regression"]
        },
        {
            "heading": "3. Unsupervised Learning",
            "subtopics": ["Clustering", "Dimensionality Reduction"]
        }
    ]