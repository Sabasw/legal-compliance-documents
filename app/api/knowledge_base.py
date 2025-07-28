#app/api/knowledge_base.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from app.utils.dependencies import get_analyzer
from app.utils.analyzer import ComplianceAnalyzer
from app.database.auth.oauth2 import get_current_user
from app.database.models.models import User
from app.config import settings
import os
import shutil
from typing import List
import logging
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.txt')

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_knowledge(
    files: List[UploadFile] = File(...),
    rebuild: bool = True,
    analyzer: ComplianceAnalyzer = Depends(get_analyzer),
    current_user: User = Depends(get_current_user)  # Requires admin privileges
):
    """Upload and process documents to update knowledge base. Admin only."""
    try:
        # analyzer = ComplianceAnalyzer()
        # Validate at least one file provided
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided for upload"
            )
        
        # Ensure knowledge directory exists
        Path(settings.KNOWLEDGE_DIR).mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        rejected_files = []
        
        for file in files:
            try:
                file_ext = Path(file.filename).suffix.lower()
                
                # Validate file type
                if file_ext not in SUPPORTED_EXTENSIONS:
                    rejected_files.append(file.filename)
                    logger.warning(f"Rejected unsupported file: {file.filename}")
                    continue
                
                # Create secure file path
                file_path = Path(settings.KNOWLEDGE_DIR) / Path(file.filename).name
                
                # Save file
                with file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                saved_files.append(str(file_path))
                logger.info(f"Saved file: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                rejected_files.append(file.filename)
            finally:
                await file.close()
        
        # If no files were successfully saved
        if not saved_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid files uploaded. Rejected: {rejected_files}"
            )
        
        # Rebuild knowledge base if requested
        rebuild_status = {}
        if rebuild:
            try:
                analyzer.rebuild_knowledge_base(saved_files)
                rebuild_status = {
                    "success": True,
                    "message": "Knowledge base rebuilt successfully",
                    "documents_processed": len(saved_files)
                }
                logger.info("Knowledge base rebuilt successfully")
            except Exception as e:
                rebuild_status = {
                    "success": False,
                    "error": str(e),
                    "documents_processed": 0
                }
                logger.error(f"Rebuild failed: {str(e)}")
        
        response = {
            "message": "File upload completed",
            "saved_files": saved_files,
            "rejected_files": rejected_files,
            "rebuild": rebuild_status if rebuild else {"message": "Rebuild not requested"}
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(e)}"
        )
@router.get("/status")
async def knowledge_status(
    analyzer: ComplianceAnalyzer = Depends(get_analyzer),
    current_user: User = Depends(get_current_user)  # Requires admin privileges
):
    """Get detailed knowledge base status. Admin only."""
    try:
        kb_status = {
            "loaded": analyzer.kb_index is not None,  # Use the property
            "chunks_count": len(analyzer.kb_chunks) if analyzer.kb_chunks else 0,  # Use the property
            "index_path": settings.FAISS_INDEX_PATH,
            "knowledge_dir": settings.KNOWLEDGE_DIR,
            "documents_count": len(os.listdir(settings.KNOWLEDGE_DIR)) if os.path.exists(settings.KNOWLEDGE_DIR) else 0
        }
        
        return kb_status
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check knowledge base status: {str(e)}"
        )

@router.post("/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_knowledge(
    analyzer: ComplianceAnalyzer = Depends(get_analyzer),
    current_user: User = Depends(get_current_user)  # Requires admin privileges
):
    """Rebuild knowledge base from existing documents with validation. Admin only."""
    try:
        # Check if knowledge directory exists
        if not os.path.exists(settings.KNOWLEDGE_DIR):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge documents directory not found"
            )
        
        # Get all supported documents - Fixed list comprehension syntax
        doc_files = [
            str(Path(settings.KNOWLEDGE_DIR) / f)  # Added closing parenthesis
            for f in os.listdir(settings.KNOWLEDGE_DIR)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not doc_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported documents found in knowledge directory"
            )
        
        try:
            analyzer.rebuild_knowledge_base(doc_files)
            return {
                "message": f"Successfully rebuilt knowledge base with {len(doc_files)} documents",
                "documents_processed": len(doc_files),
                "document_paths": doc_files
            }
            
        except Exception as e:
            logger.error(f"Rebuild failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Knowledge base rebuild failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during rebuild"
        )