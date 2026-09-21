from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import get_rag_response

app = FastAPI(title="Amazon Policy RAG Assistant")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Amazon Policy RAG Assistant is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    result = get_rag_response(request.question)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"]
    }