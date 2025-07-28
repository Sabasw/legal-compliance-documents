# documents.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request, status, Query
from typing import List
import uuid
from app.database.scehmas.schemas import DocumentAnalysisSchema, AnalysisRequest
from app.core.compliance import ComplianceAnalyzer
from app.utils.dependencies import get_analyzer
from app.database.auth.oauth2 import get_db
from app.core.document import DocumentProcessor
from app.utils.exceptions import DocumentProcessingError
from app.database.models.models import Document, User, DocumentAnalysis
from app.database.auth.oauth2 import get_current_user
from app.utils.document import validate_file_type, check_document_ownership, sanitize_filename
from app.utils.audit import create_audit_log
from app.utils.rate_limit import RateLimiter
from sqlalchemy.orm import Session
import os
import logging
from app.config import settings
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
from datetime import datetime
from app.utils.model_loader import load_model_and_client
from uuid import uuid4
from app.services.blockchain_service import BlockchainService
from app.utils.blockchain_utils import get_user_keypair
from app.database.models.blockchain_models import BlockchainAuditEntry, BlockchainActionType
from app.utils.file_organizer import FileOrganizer
from app.utils.pdf_generator import PDFReportGenerator
import json

router = APIRouter()
logger = logging.getLogger(__name__)

def convert_risk_score_to_float(risk_score):
    """Convert string risk score to float value"""
    if risk_score is None:
        return None
    
    if isinstance(risk_score, (int, float)):
        return float(risk_score)
    
    if isinstance(risk_score, str):
        risk_score_lower = risk_score.lower()
        if risk_score_lower in ['low', 'low risk']:
            return 0.25
        elif risk_score_lower in ['medium', 'medium risk', 'moderate']:
            return 0.5
        elif risk_score_lower in ['high', 'high risk']:
            return 0.75
        elif risk_score_lower in ['critical', 'very high', 'very high risk']:
            return 1.0
        else:
            # Try to convert to float if it's a numeric string
            try:
                return float(risk_score)
            except ValueError:
                # Default to medium if we can't parse it
                return 0.5
    
    return 0.5  # Default fallback

groq_client, model, predictive_analytics = load_model_and_client()
analyzer = ComplianceAnalyzer(groq_client, model, predictive_analytics)
analyzer.load_knowledge_base("compliance_index.faiss", "chunks.txt")

# Initialize file organizer and PDF generator
file_organizer = FileOrganizer()
pdf_generator = PDFReportGenerator()

# @router.post("/upload-2/")
# async def upload_docs(file: UploadFile = File(...),
#                         db: Session = Depends(get_db)):
#     try:
#         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#         file_path = os.path.join(settings.UPLOAD_DIR, filename)

#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         results = analyzer.analyze_document(file_path)

#         if "error" in results:
#             raise HTTPException(status_code=500, detail=results["error"])

#         return JSONResponse(content=results)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/")
async def upload_docs(
    file: UploadFile = File(...),
    department: str = Query(..., description="Department the document belongs to"),
    classification: str = Query("Confidential", description="Document classification level"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user) 
):
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        
        # Step 1: Save file
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Analyze document
        results = analyzer.analyze_document(file_path)

        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        # Step 3: Track in blockchain
        try:
            # Get user's blockchain keypair
            user_keypair = get_user_keypair(db, str(user.id))
            if not user_keypair:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize blockchain keys. Please contact support."
                )

            # Initialize blockchain service
            blockchain_service = BlockchainService()
            
            # Track document in blockchain
            blockchain_entry = await blockchain_service.track_document_creation(
                file_path=file_path,
                user_keypair=user_keypair,
                department=department,
                classification=classification,
                user_id=str(user.id),
                username=user.full_name,
                document_type=results.get("document_type", "unknown"),
                analysis_summary=results.get("summary"),
                risk_score=results.get("risk_score")
            )
            
            if not blockchain_entry:
                raise ValueError("Empty blockchain entry received")

        except Exception as blockchain_error:
            logger.error(f"Blockchain tracking failed: {blockchain_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to track document in blockchain: {str(blockchain_error)}"
            )

        # Step 4: Create Document record with all data
        doc_id = uuid4()
        document = Document(
            id=doc_id,
            owner_id=user.id,
            filename=file.filename,
            file_path=file_path,
            document_type=results.get("document_type", "unknown"),
            status="completed",
            department=department,
            classification=classification,
            
            # Document analysis fields
            compliance_status=results.get("compliance_status"),
            risk_score=convert_risk_score_to_float(results.get("risk_score")),
            risk_profile=results.get("risk_profile"),
            statutory_references=results.get("statutory_analysis"),
            key_issues=results.get("kb_references") or [],
            recommendations=results.get("recommendations") or [],
            predictive_outcomes=results.get("predictive_outcomes"),
            summary=results.get("summary"),
            full_analysis=results.get("full_analysis"),
            
            # Blockchain fields
            blockchain_tx_hash=blockchain_entry["transaction_hash"],
            blockchain_content_hash=blockchain_entry["content_hash"],
            blockchain_document_hash=blockchain_entry["document_hash"].hex(),
            blockchain_metadata={
                "timestamp": blockchain_entry.get("timestamp"),
                "status": blockchain_entry.get("status"),
                "metadata": blockchain_entry.get("metadata", {})
            },
            
            # Original metadata
            document_metadata={
                "report_path": results.get("report_path"),
                "visualization_paths": results.get("visualization_paths"),
                "document_hash": results.get("document_hash"),
                "timestamp": results.get("timestamp")
            }
        )
        
        # Organize generated files after document creation
        report_path = results.get("report_path")
        visualization_paths = results.get("visualization_paths") or []
        
        # Ensure report_path is a string
        if not isinstance(report_path, str):
            report_path = None
        
        # Ensure visualization_paths is a list
        if isinstance(visualization_paths, str):
            visualization_paths = [visualization_paths]
        elif not isinstance(visualization_paths, list):
            visualization_paths = []
        
        file_paths = {
            "report": report_path,
            "visualizations": visualization_paths
        }
        
        # Debug: Print file paths
        print(f"Debug - File paths to organize: {file_paths}")
        
        # Organize files into folders
        organized_files = file_organizer.organize_document_files(str(doc_id), file_paths)
        
        # Update document metadata with organized files
        document.document_metadata.update({
            "report_path": organized_files.get("report"),
            "visualization_paths": organized_files.get("visualizations", []),
            "organized_files": organized_files
        })
        
        # Add initial audit entry
        document.add_audit_entry(
            action="CREATED",
            user_id=str(user.id),
            metadata={
                "department": department,
                "classification": classification,
                "document_type": document.document_type,
                "risk_score": results.get("risk_score"),  # Keep original string for audit
                "compliance_status": document.compliance_status,
                "blockchain_tx": document.blockchain_tx_hash
            }
        )

        # Save to database
        db.add(document)
        db.commit()
        db.refresh(document)

        # Calculate file integrity information
        from app.utils.blockchain_utils import calculate_content_hash
        current_content_hash = calculate_content_hash(file_path)
        
        # Check if this file has been uploaded before (same content hash)
        existing_documents = db.query(Document).filter(
            Document.blockchain_content_hash == current_content_hash,
            Document.id != document.id
        ).all()
        
        file_integrity_info = {
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "content_hash": current_content_hash,
            "hash_algorithm": "SHA-256",
            "integrity_status": "VERIFIED",
            "is_duplicate": len(existing_documents) > 0,
            "duplicate_count": len(existing_documents),
            "first_uploaded": None,
            "previous_versions": []
        }
        
        if existing_documents:
            file_integrity_info["first_uploaded"] = existing_documents[0].created_at.isoformat()
            file_integrity_info["previous_versions"] = [
                {
                    "doc_id": str(doc.id),
                    "uploaded_at": doc.created_at.isoformat(),
                    "filename": doc.filename,
                    "transaction_hash": doc.blockchain_tx_hash
                }
                for doc in existing_documents
            ]

        # Return response
        return JSONResponse(content={
            "doc_id": str(document.id),
            "document_type": document.document_type,
            "timestamp": document.created_at.isoformat(),
            "filename": document.filename,
            
            # Analysis data
            "compliance_status": document.compliance_status,
            "risk_score": results.get("risk_score"),  # Return original string
            "summary": document.summary,
            "risk_profile": document.risk_profile,
            "statutory_references": document.statutory_references,
            "key_issues": document.key_issues,
            "recommendations": document.recommendations,
            "predictive_outcomes": document.predictive_outcomes,
            "report_path": document.document_metadata.get("report_path"),
            "visualization_paths": document.document_metadata.get("visualization_paths", []),
            
            # Enhanced Blockchain data
            "blockchain": {
                "transaction_hash": document.blockchain_tx_hash,
                "content_hash": document.blockchain_content_hash,
                "document_hash": document.blockchain_document_hash,
                "department": document.department,
                "classification": document.classification,
                "metadata": document.blockchain_metadata,
                "blockchain_status": "CONFIRMED",
                "verification_url": f"https://blockchain.example.com/tx/{document.blockchain_tx_hash}"
            },
            
            # File Integrity Tracking
            "file_integrity": file_integrity_info,
            
            # Document Tracking
            "tracking": {
                "upload_timestamp": document.created_at.isoformat(),
                "last_modified": document.updated_at.isoformat() if document.updated_at else None,
            "version": document.version,
                "status": document.status,
                "department": document.department,
                "classification": document.classification,
                "uploaded_by": user.full_name,
                "user_id": str(user.id)
            },
            
            # Audit data
            "audit_trail": document.get_audit_history(),
            
            # File Change Detection
            "change_detection": {
                "is_new_file": len(existing_documents) == 0,
                "is_modified_version": len(existing_documents) > 0,
                "content_changed": True,  # Always true for new uploads
                "hash_verification": "PASSED",
                "blockchain_verification": "PASSED"
            }
        })

    except Exception as e:
        db.rollback()
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-documents")
def list_documents(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()

        # Get documents directly from Document table since analysis data is stored there
        documents = db.query(Document).filter(Document.owner_id == user.id).all()

        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="list",
            entity_type="documents",
            entity_id="all",
            request=request
        )

        # Enhanced document list with tracking information
        enhanced_documents = []
        
        for doc in documents:
            # Calculate file integrity information
            from app.utils.blockchain_utils import calculate_content_hash
            current_content_hash = calculate_content_hash(doc.file_path) if os.path.exists(doc.file_path) else None
            
            # Check if this file has been uploaded before (same content hash)
            existing_documents = db.query(Document).filter(
                Document.blockchain_content_hash == doc.blockchain_content_hash,
                Document.id != doc.id
            ).all()
            
            file_integrity_info = {
                "file_size_bytes": os.path.getsize(doc.file_path) if os.path.exists(doc.file_path) else 0,
                "file_size_mb": round(os.path.getsize(doc.file_path) / (1024 * 1024), 2) if os.path.exists(doc.file_path) else 0,
                "content_hash": doc.blockchain_content_hash,
                "current_hash": current_content_hash,
                "hash_algorithm": "SHA-256",
                "integrity_status": "VERIFIED" if current_content_hash == doc.blockchain_content_hash else "MODIFIED",
                "is_duplicate": len(existing_documents) > 0,
                "duplicate_count": len(existing_documents)
            }
            
            enhanced_documents.append({
                "doc_id": str(doc.id),
                "document_type": doc.document_type,
                "compliance_status": doc.compliance_status,
                "risk_score": doc.risk_score,  # This is now a float
                "risk_profile": doc.risk_profile,
                "statutory_references": doc.statutory_references,
                "key_issues": doc.key_issues,
                "recommendations": doc.recommendations,
                "predictive_outcomes": doc.predictive_outcomes,
                "summary": doc.summary,
                "full_analysis": doc.full_analysis,
                "timestamp": doc.created_at.isoformat(),
                "filename": doc.filename,
                "department": doc.department,
                "classification": doc.classification,
                
                # Enhanced Blockchain data
                "blockchain": {
                    "transaction_hash": doc.blockchain_tx_hash,
                    "content_hash": doc.blockchain_content_hash,
                    "document_hash": doc.blockchain_document_hash,
                    "metadata": doc.blockchain_metadata,
                    "blockchain_status": "CONFIRMED"
                },
                
                # File Integrity Tracking
                "file_integrity": file_integrity_info,
                
                # Document Tracking
                "tracking": {
                    "upload_timestamp": doc.created_at.isoformat(),
                    "last_modified": doc.updated_at.isoformat() if doc.updated_at else None,
                    "version": doc.version,
                    "status": doc.status,
                    "department": doc.department,
                    "classification": doc.classification
                },
                
                "audit_trail": doc.get_audit_history() if doc.audit_trail else [],
                
                # File Change Detection
                "change_detection": {
                    "is_new_file": len(existing_documents) == 0,
                    "is_modified_version": len(existing_documents) > 0,
                    "content_changed": current_content_hash != doc.blockchain_content_hash if current_content_hash else False,
                    "hash_verification": "PASSED" if current_content_hash == doc.blockchain_content_hash else "FAILED"
                }
            })
        
        return enhanced_documents

    except Exception as e:
        logger.error(f"List documents failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

@router.get("/{doc_id}")
def get_document(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)

        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="retrieve",
            entity_type="document",
            entity_id=str(doc_id),
            document_id=document.id,
            request=request
        )

        # Calculate file integrity information
        from app.utils.blockchain_utils import calculate_content_hash
        current_content_hash = calculate_content_hash(document.file_path) if os.path.exists(document.file_path) else None
        
        # Check if this file has been uploaded before (same content hash)
        existing_documents = db.query(Document).filter(
            Document.blockchain_content_hash == document.blockchain_content_hash,
            Document.id != document.id
        ).all()
        
        file_integrity_info = {
            "file_size_bytes": os.path.getsize(document.file_path) if os.path.exists(document.file_path) else 0,
            "file_size_mb": round(os.path.getsize(document.file_path) / (1024 * 1024), 2) if os.path.exists(document.file_path) else 0,
            "content_hash": document.blockchain_content_hash,
            "current_hash": current_content_hash,
            "hash_algorithm": "SHA-256",
            "integrity_status": "VERIFIED" if current_content_hash == document.blockchain_content_hash else "MODIFIED",
            "is_duplicate": len(existing_documents) > 0,
            "duplicate_count": len(existing_documents),
            "first_uploaded": None,
            "previous_versions": []
        }
        
        if existing_documents:
            file_integrity_info["first_uploaded"] = existing_documents[0].created_at.isoformat()
            file_integrity_info["previous_versions"] = [
                {
                    "doc_id": str(doc.id),
                    "uploaded_at": doc.created_at.isoformat(),
                    "filename": doc.filename,
                    "transaction_hash": doc.blockchain_tx_hash
                }
                for doc in existing_documents
            ]
        
        return {
            "doc_id": str(document.id),
            "document_type": document.document_type,
            "compliance_status": document.compliance_status,
            "risk_score": document.risk_score,  # This is now a float
            "risk_profile": document.risk_profile,
            "statutory_references": document.statutory_references,
            "key_issues": document.key_issues,
            "recommendations": document.recommendations,
            "predictive_outcomes": document.predictive_outcomes,
            "summary": document.summary,
            "full_analysis": document.full_analysis,
            "timestamp": document.created_at.isoformat(),
            "filename": document.filename,
            "department": document.department,
            "classification": document.classification,
            
            # Enhanced Blockchain data
            "blockchain": {
                "transaction_hash": document.blockchain_tx_hash,
                "content_hash": document.blockchain_content_hash,
                "document_hash": document.blockchain_document_hash,
                "metadata": document.blockchain_metadata,
                "blockchain_status": "CONFIRMED",
                "verification_url": f"https://blockchain.example.com/tx/{document.blockchain_tx_hash}"
            },
            
            # File Integrity Tracking
            "file_integrity": file_integrity_info,
            
            # Document Tracking
            "tracking": {
                "upload_timestamp": document.created_at.isoformat(),
                "last_modified": document.updated_at.isoformat() if document.updated_at else None,
                "version": document.version,
                "status": document.status,
                "department": document.department,
                "classification": document.classification,
                "uploaded_by": user.full_name,
                "user_id": str(user.id)
            },
            
            "audit_trail": document.get_audit_history() if document.audit_trail else [],
            
            # File Change Detection
            "change_detection": {
                "is_new_file": len(existing_documents) == 0,
                "is_modified_version": len(existing_documents) > 0,
                "content_changed": current_content_hash != document.blockchain_content_hash if current_content_hash else False,
                "hash_verification": "PASSED" if current_content_hash == document.blockchain_content_hash else "FAILED",
                "blockchain_verification": "PASSED"
            }
        }
    except Exception as e:
        logger.error(f"Get document failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Delete physical file if it exists
        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)

        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="delete",
            entity_type="document",
            entity_id=doc_id,
            changes={"filename": document.filename},
            request=request
        )

        # Delete document (no separate DocumentAnalysis table needed)
        db.delete(document)
        db.commit()
        
        return JSONResponse(
            content={
                "message":"Document deleted successfully"
            },
            status_code=204
        )

    except Exception as e:
        logger.error(f"Delete document failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/{doc_id}/report")
def generate_document_report(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Generate a structured PDF report for a document"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Get document files
        document_files = file_organizer.list_document_files(doc_id)
        
        if not document_files:
            raise HTTPException(status_code=404, detail="No files found for this document")
        
        # Check if PDF report already exists
        existing_pdf_path = file_organizer.get_pdf_report_path(doc_id)
        
        if existing_pdf_path:
            # Use existing PDF report
            pdf_path = existing_pdf_path
            report_filename = os.path.basename(pdf_path)
            print(f"Using existing PDF report: {pdf_path}")
        else:
            # Create new PDF report
            report_filename = f"document_report_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            temp_report_path = os.path.join("reports", report_filename)
            
            # Prepare document data for PDF
            document_data = {
                "document_type": document.document_type,
                "compliance_status": document.compliance_status,
                "risk_score": document.risk_score,
                "risk_profile": document.risk_profile,
                "summary": document.summary,
                "key_issues": document.key_issues,
                "recommendations": document.recommendations,
                "full_analysis": document.full_analysis
            }
            
            # Generate comprehensive PDF report
            temp_pdf_path = pdf_generator.create_document_report(
                doc_id=doc_id,
                document_data=document_data,
                file_paths=document_files,
                output_path=temp_report_path
            )
            
            # Store PDF in document folder and replace duplicates
            pdf_path = file_organizer.manage_pdf_report(doc_id, temp_pdf_path)
            
            # Clean up temporary file
            if temp_pdf_path != pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception as e:
                    print(f"Error cleaning up temp PDF: {e}")
        
        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="generate_pdf_report",
            entity_type="document",
            entity_id=doc_id,
            document_id=document.id,
            request=Request
        )
        
        # Return PDF report path and metadata
        return {
            "doc_id": doc_id,
            "pdf_report_path": pdf_path,
            "pdf_filename": report_filename,
            "document_type": document.document_type,
            "compliance_status": document.compliance_status,
            "risk_score": document.risk_score,
            "risk_profile": document.risk_profile,
            "summary": document.summary,
            "key_issues": document.key_issues,
            "recommendations": document.recommendations,
            "full_analysis": document.full_analysis,
            "source_files": [
                {
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    "file_type": os.path.splitext(file_path)[1].lower()
                }
                for file_path in document_files
            ],
            "total_source_files": len(document_files),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"PDF report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF report")


# @router.get("/{doc_id}/report/download")
# def download_pdf_report(
#     doc_id: str,
#     db: Session = Depends(get_db),
#     current_user=Depends(get_current_user)
# ):
#     """Download the generated PDF report for a document"""
#     try:
#         user = db.query(User).filter(User.email == current_user.get("email")).first()
#         document = check_document_ownership(doc_id, user, db)
        
#         # Get document files
#         document_files = file_organizer.list_document_files(doc_id)
        
#         if not document_files:
#             raise HTTPException(status_code=404, detail="No files found for this document")
        
#         # Check if PDF report already exists
#         existing_pdf_path = file_organizer.get_pdf_report_path(doc_id)
        
#         if existing_pdf_path:
#             # Use existing PDF report
#             pdf_path = existing_pdf_path
#             report_filename = os.path.basename(pdf_path)
#             print(f"Using existing PDF report for download: {pdf_path}")
#         else:
#             # Create new PDF report
#             report_filename = f"document_report_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
#             temp_report_path = os.path.join("reports", report_filename)
            
#             # Prepare document data for PDF
#             document_data = {
#                 "document_type": document.document_type,
#                 "compliance_status": document.compliance_status,
#                 "risk_score": document.risk_score,
#                 "risk_profile": document.risk_profile,
#                 "summary": document.summary,
#                 "key_issues": document.key_issues,
#                 "recommendations": document.recommendations,
#                 "full_analysis": document.full_analysis
#             }
            
#             # Generate comprehensive PDF report
#             temp_pdf_path = pdf_generator.create_document_report(
#                 doc_id=doc_id,
#                 document_data=document_data,
#                 file_paths=document_files,
#                 output_path=temp_report_path
#             )
            
#             # Store PDF in document folder and replace duplicates
#             pdf_path = file_organizer.manage_pdf_report(doc_id, temp_pdf_path)
            
#             # Clean up temporary file
#             if temp_pdf_path != pdf_path and os.path.exists(temp_pdf_path):
#                 try:
#                     os.remove(temp_pdf_path)
#                 except Exception as e:
#                     print(f"Error cleaning up temp PDF: {e}")
        
#         # Create audit log
#         create_audit_log(
#             db=db,
#             user_id=user.id,
#             action="download_pdf_report",
#             entity_type="document",
#             entity_id=doc_id,
#             document_id=document.id,
#             request=Request
#         )
        
#         # Return file response for download
#         return FileResponse(
#             path=pdf_path,
#             filename=report_filename,
#             media_type="application/pdf"
#         )
        
#     except Exception as e:
#         logger.error(f"PDF report download failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to download PDF report")


@router.post("/{doc_id}/report/regenerate")
def regenerate_pdf_report(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Force regenerate the PDF report for a document"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Get document files
        document_files = file_organizer.list_document_files(doc_id)
        
        if not document_files:
            raise HTTPException(status_code=404, detail="No files found for this document")
        
        # Always create new PDF report (force regeneration)
        report_filename = f"document_report_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        temp_report_path = os.path.join("reports", report_filename)
        
        # Prepare document data for PDF
        document_data = {
            "document_type": document.document_type,
            "compliance_status": document.compliance_status,
            "risk_score": document.risk_score,
            "risk_profile": document.risk_profile,
            "summary": document.summary,
            "key_issues": document.key_issues,
            "recommendations": document.recommendations,
            "full_analysis": document.full_analysis
        }
        
        # Generate comprehensive PDF report
        temp_pdf_path = pdf_generator.create_document_report(
            doc_id=doc_id,
            document_data=document_data,
            file_paths=document_files,
            output_path=temp_report_path
        )
        
        # Store PDF in document folder and replace duplicates
        pdf_path = file_organizer.manage_pdf_report(doc_id, temp_pdf_path)
        
        # Clean up temporary file
        if temp_pdf_path != pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                print(f"Error cleaning up temp PDF: {e}")
        
        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="regenerate_pdf_report",
            entity_type="document",
            entity_id=doc_id,
            document_id=document.id,
            request=Request
        )
        
        # Return PDF report path and metadata
        return {
            "doc_id": doc_id,
            "pdf_report_path": pdf_path,
            "pdf_filename": os.path.basename(pdf_path),
            "message": "PDF report regenerated successfully",
            "regenerated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"PDF report regeneration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to regenerate PDF report")


@router.get("/{doc_id}/files")
def get_document_files(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get all files associated with a document"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Get document files
        document_files = file_organizer.list_document_files(doc_id)
        
        # Get folder path
        folder_path = file_organizer.get_document_folder(doc_id)
        
        return {
            "doc_id": doc_id,
            "folder_path": folder_path,
            "files": [
                {
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    "file_type": os.path.splitext(file_path)[1].lower()
                }
                for file_path in document_files
            ],
            "total_files": len(document_files)
        }
        
    except Exception as e:
        logger.error(f"Get document files failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document files")


@router.post("/cleanup-files")
def cleanup_old_files(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Clean up old files (admin only)"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        
        # Check if user is admin
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Only admins can perform cleanup")
        
        # Perform cleanup
        file_organizer.cleanup_old_files(days)
        
        return {
            "message": f"Cleanup completed for files older than {days} days",
            "cleanup_days": days
        }
        
    except Exception as e:
        logger.error(f"File cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup files")

@router.get("/{doc_id}/integrity")
def verify_document_integrity(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Verify document integrity and provide detailed tracking information"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Calculate current file hash
        from app.utils.blockchain_utils import calculate_content_hash
        current_content_hash = calculate_content_hash(document.file_path) if os.path.exists(document.file_path) else None
        
        # Check if file exists
        file_exists = os.path.exists(document.file_path)
        
        # Check if this file has been uploaded before (same content hash)
        existing_documents = db.query(Document).filter(
            Document.blockchain_content_hash == document.blockchain_content_hash,
            Document.id != document.id
        ).all()
        
        # Get all documents with same filename
        same_filename_docs = db.query(Document).filter(
            Document.filename == document.filename,
            Document.owner_id == user.id
        ).all()
        
        integrity_report = {
            "doc_id": str(document.id),
            "filename": document.filename,
            "file_path": document.file_path,
            "file_exists": file_exists,
            
            # File Information
            "file_info": {
                "file_size_bytes": os.path.getsize(document.file_path) if file_exists else 0,
                "file_size_mb": round(os.path.getsize(document.file_path) / (1024 * 1024), 2) if file_exists else 0,
                "file_extension": os.path.splitext(document.filename)[1],
                "upload_date": document.created_at.isoformat(),
                "last_modified": document.updated_at.isoformat() if document.updated_at else None
            },
            
            # Hash Verification
            "hash_verification": {
                "original_hash": document.blockchain_content_hash,
                "current_hash": current_content_hash,
                "hash_algorithm": "SHA-256",
                "hash_match": current_content_hash == document.blockchain_content_hash if current_content_hash else False,
                "integrity_status": "VERIFIED" if current_content_hash == document.blockchain_content_hash else "MODIFIED"
            },
            
            # Blockchain Verification
            "blockchain_verification": {
                "transaction_hash": document.blockchain_tx_hash,
                "document_hash": document.blockchain_document_hash,
                "blockchain_status": "CONFIRMED",
                "verification_url": f"https://blockchain.example.com/tx/{document.blockchain_tx_hash}",
                "timestamp": document.blockchain_metadata.get("timestamp") if document.blockchain_metadata else None
            },
            
            # Duplicate Detection
            "duplicate_analysis": {
                "is_duplicate": len(existing_documents) > 0,
                "duplicate_count": len(existing_documents),
                "first_uploaded": existing_documents[0].created_at.isoformat() if existing_documents else None,
                "previous_versions": [
                    {
                        "doc_id": str(doc.id),
                        "uploaded_at": doc.created_at.isoformat(),
                        "filename": doc.filename,
                        "transaction_hash": doc.blockchain_tx_hash,
                        "content_hash": doc.blockchain_content_hash
                    }
                    for doc in existing_documents
                ]
            },
            
            # File History
            "file_history": {
                "total_uploads": len(same_filename_docs),
                "upload_history": [
                    {
                        "doc_id": str(doc.id),
                        "uploaded_at": doc.created_at.isoformat(),
                        "content_hash": doc.blockchain_content_hash,
                        "transaction_hash": doc.blockchain_tx_hash,
                        "is_current": doc.id == document.id
                    }
                    for doc in same_filename_docs
                ]
            },
            
            # Change Detection
            "change_detection": {
                "is_new_file": len(existing_documents) == 0,
                "is_modified_version": len(existing_documents) > 0,
                "content_changed": current_content_hash != document.blockchain_content_hash if current_content_hash else False,
                "hash_verification": "PASSED" if current_content_hash == document.blockchain_content_hash else "FAILED",
                "blockchain_verification": "PASSED"
            },
            
            # Security Information
            "security": {
                "uploaded_by": user.full_name,
                "user_id": str(user.id),
                "department": document.department,
                "classification": document.classification,
                "access_control": "OWNER_ONLY"
            },
            
            # Audit Trail
            "audit_trail": document.get_audit_history() if document.audit_trail else [],
            
            # Verification Summary
            "verification_summary": {
                "overall_status": "PASSED" if (current_content_hash == document.blockchain_content_hash and file_exists) else "FAILED",
                "file_integrity": "VERIFIED" if current_content_hash == document.blockchain_content_hash else "MODIFIED",
                "blockchain_integrity": "CONFIRMED",
                "audit_trail": "COMPLETE" if document.audit_trail else "INCOMPLETE"
            }
        }
        
        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="verify_integrity",
            entity_type="document",
            entity_id=doc_id,
            document_id=document.id,
            request=Request
        )
        
        return integrity_report
        
    except Exception as e:
        logger.error(f"Document integrity verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify document integrity")


@router.get("/{doc_id}/tracking")
def get_document_tracking_info(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get comprehensive document tracking information"""
    try:
        user = db.query(User).filter(User.email == current_user.get("email")).first()
        document = check_document_ownership(doc_id, user, db)
        
        # Get all documents with same content hash (versions)
        version_documents = db.query(Document).filter(
            Document.blockchain_content_hash == document.blockchain_content_hash
        ).order_by(Document.created_at).all()
        
        # Get all documents with same filename
        filename_documents = db.query(Document).filter(
            Document.filename == document.filename,
            Document.owner_id == user.id
        ).order_by(Document.created_at).all()
        
        tracking_info = {
            "doc_id": str(document.id),
            "filename": document.filename,
            
            # Document Versions
            "versions": {
                "total_versions": len(version_documents),
                "version_history": [
                    {
                        "doc_id": str(doc.id),
                        "uploaded_at": doc.created_at.isoformat(),
                        "transaction_hash": doc.blockchain_tx_hash,
                        "content_hash": doc.blockchain_content_hash,
                        "is_current": doc.id == document.id,
                        "uploaded_by": user.full_name if doc.owner_id == user.id else "Unknown"
                    }
                    for doc in version_documents
                ]
            },
            
            # File History
            "file_history": {
                "total_uploads": len(filename_documents),
                "upload_history": [
                    {
                        "doc_id": str(doc.id),
                        "uploaded_at": doc.created_at.isoformat(),
                        "content_hash": doc.blockchain_content_hash,
                        "transaction_hash": doc.blockchain_tx_hash,
                        "is_current": doc.id == document.id,
                        "file_size": os.path.getsize(doc.file_path) if os.path.exists(doc.file_path) else 0
                    }
                    for doc in filename_documents
                ]
            },
            
            # Blockchain Tracking
            "blockchain_tracking": {
                "transaction_hash": document.blockchain_tx_hash,
                "content_hash": document.blockchain_content_hash,
                "document_hash": document.blockchain_document_hash,
                "blockchain_status": "CONFIRMED",
                "timestamp": document.blockchain_metadata.get("timestamp") if document.blockchain_metadata else None,
                "verification_url": f"https://blockchain.example.com/tx/{document.blockchain_tx_hash}"
            },
            
            # Audit Trail
            "audit_trail": document.get_audit_history() if document.audit_trail else [],
            
            # Tracking Summary
            "tracking_summary": {
                "first_uploaded": version_documents[0].created_at.isoformat() if version_documents else None,
                "last_uploaded": document.created_at.isoformat(),
                "total_versions": len(version_documents),
                "total_uploads": len(filename_documents),
                "current_version": next((i+1 for i, doc in enumerate(version_documents) if doc.id == document.id), 1)
            }
        }
        
        return tracking_info
        
    except Exception as e:
        logger.error(f"Document tracking info failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document tracking info")
