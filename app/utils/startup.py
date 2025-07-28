# app/startup.py
from app.core.compliance import ComplianceAnalyzer, PredictiveAnalytics
from app.utils.model_loader import load_model_and_client
from sentence_transformers import SentenceTransformer
from groq import Groq

groq_client: Groq = None
model: SentenceTransformer = None
predictive: PredictiveAnalytics = None
analyzer: ComplianceAnalyzer = None

def initialize_global_analyzer():
    global groq_client, model, predictive, analyzer

    # Your init logic (use your own config if needed)
    groq_client, model, predictive = load_model_and_client()
    analyzer = ComplianceAnalyzer(groq_client, model, predictive)
    analyzer.load_knowledge_base("compliance_index.faiss", "chunks.txt")
