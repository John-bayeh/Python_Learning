from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ai_service import ask_ai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Question(BaseModel):
    text: str
    history: Optional[List[Message]] = []

@app.get("/")
def root():
    return {"status": "AI is running"}

@app.post("/ask")
def ask(question: Question):
    answer = ask_ai(question.text, [msg.dict() for msg in question.history])
    return {"question": question.text, "answer": answer}
@app.post("/analyze-sentiment")
def analyze_sentiment_endpoint(question: Question):
    # analyze multiple feedbacks at once
    answer = ask_ai(f"Analyze sentiment of this feedback: {question.text}")
    return {"feedback": question.text, "analysis": answer}