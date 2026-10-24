from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask_rag

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/pergunta")
def ask_question(data: Question):
    return ask_rag(data.question)
