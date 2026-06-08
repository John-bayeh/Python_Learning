from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← add this import
from pydantic import BaseModel
from ai_service import ask_ai

app = FastAPI()

# ← this entire block is missing from your code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "AI is running"}

@app.post("/ask")
def ask(question: Question):
    answer = ask_ai(question.text)
    return {"question": question.text, "answer": answer}