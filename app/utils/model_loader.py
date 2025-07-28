# app/utils/model_loader.py
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.utils.analyzer import PredictiveAnalytics
from app.config import settings

def load_model_and_client():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groq_client = Groq(api_key=settings.GROQ_API_KEY)
    try:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    except Exception:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

    predictive = PredictiveAnalytics(model)
    return groq_client, model, predictive
