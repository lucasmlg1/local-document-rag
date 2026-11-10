from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask_rag, warmup
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()  
    yield

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/pergunta")
def ask_question(data: Question):
    return ask_rag(data.question)
