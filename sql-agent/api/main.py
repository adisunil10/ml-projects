import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run

app = FastAPI(title="SQL Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory=str(Path(__file__).parent.parent / "frontend"), html=True), name="frontend")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return run(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
