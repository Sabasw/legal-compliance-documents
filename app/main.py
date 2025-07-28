from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from app.database.db.db_connection import engine, Base, get_db, create_tables
from fastapi.middleware.cors import CORSMiddleware
import warnings
from app.database.models import *
from app.api import auth, knowledge_base, documents, blockchain
from app.api.blockchain import router as blockchain_router
from app.utils.startup import initialize_global_analyzer

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

app = FastAPI()


print("backend start")
# Enable CORS for specific origins or allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    # Import models
    
    from app.database.models.models import (
            User, Document, ComplianceRule, DocumentAnalysis, RiskScore,
            AuditLog
        )
    
    create_tables()
    
    print("✅ Database initialized on startup")
    initialize_global_analyzer()

@app.get("/")
async def root():
    return {"message": "Welcome to the AI powered Dashboard Backend"}

app.include_router(auth.router)
app.include_router(knowledge_base.router, tags=["knowledge_base"])
app.include_router(documents.router, tags=["documents"])
# app.include_router(blockchain_router, tags=["blockchain"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
