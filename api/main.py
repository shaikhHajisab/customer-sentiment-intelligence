import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import PredictRequest, BatchPredictRequest, PredictResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# Global model variables
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading pre-trained model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    yield
    print("Shutting down model...")

app = FastAPI(
    title="Customer Sentiment Intelligence API",
    description="Production-grade API for text sentiment classification using pre-trained RoBERTa.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "framework": "FastAPI + Transformers"
    }

def get_prediction(text: str) -> PredictResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    confidence, class_idx = torch.max(probs, dim=0)
    
    label = LABELS[class_idx.item()]
    
    return PredictResponse(
        text=text,
        label=label,
        confidence=round(confidence.item(), 4)
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment for a single text input."""
    return get_prediction(request.text)

@app.post("/batch_predict", response_model=list[PredictResponse])
def batch_predict(request: BatchPredictRequest):
    """Predict sentiment for a batch of text inputs."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="List of texts cannot be empty.")
    
    # Sequential inference for simplicity, can be batched for high throughput
    results = [get_prediction(text) for text in request.texts]
    return results
