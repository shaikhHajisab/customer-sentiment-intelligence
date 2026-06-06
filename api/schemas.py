from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
