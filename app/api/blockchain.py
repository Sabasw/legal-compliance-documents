from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from app.database.auth.oauth2 import get_current_user
from app.database.db.db_connection import get_db
from app.database.models.blockchain_models import BlockchainDocument, BlockchainAuditEntry, BlockchainUserKeys, BlockchainActionType
from app.database.scehmas.blockchain_schemas import (
    BlockchainDocumentCreate,
    BlockchainDocumentResponse,
    BlockchainAuditEntryCreate,
    BlockchainAuditEntryResponse,
    BlockchainDocumentHistory
)
from app.services.blockchain_service import BlockchainService
from solders.keypair import Keypair
import os
from datetime import datetime
from app.database.models.models import User
from uuid import UUID
from app.utils.blockchain_utils import get_user_keypair
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blockchain", tags=["blockchain"])
blockchain_service = BlockchainService()

@router.post("/documents/", response_model=BlockchainDocumentResponse)
async def create_blockchain_document(
    user_data: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    department: str = Query(..., description="Department the document belongs to"),  # Make department required
    classification: str = Query("Confidential", description="Document classification level")
):
    """Create a new document with blockchain tracking"""
    file_path = None
    try:
        # Get user from database
        user = db.query(User).filter(User.email == user_data["email"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Save file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get user's blockchain key
        blockchain_key = db.query(BlockchainUserKeys).filter_by(user_id=user.id).first()
        if not blockchain_key:
            logger.warning(f"No blockchain keys found for user {user.id}, attempting to create")

        # Get Solana keypair using the utility function
        user_keypair = get_user_keypair(db, str(user.id))
        if not user_keypair:
            logger.error(f"Failed to get keypair for user {user.id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize blockchain keys. Please contact support."
            )

        # Track in blockchain
        try:
            blockchain_entry = await blockchain_service.track_document_creation(
                file_path=file_path,
                user_keypair=user_keypair,
                department=department,  # Use the required department parameter
                classification=classification,
                user_id=str(user.id),
                username=user.full_name
            )
            
            logger.debug(f"Blockchain entry received: {blockchain_entry}")
            
            if not blockchain_entry:
                raise ValueError("Empty blockchain entry received")
                
            # Validate blockchain entry structure
            required_fields = ["document_hash", "content_hash", "transaction_hash"]
            missing_fields = [field for field in required_fields if field not in blockchain_entry]
            if missing_fields:
                raise ValueError(f"Missing required fields in blockchain entry: {missing_fields}")

        except Exception as blockchain_error:
            logger.error(f"Blockchain tracking failed: {blockchain_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to track document in blockchain: {str(blockchain_error)}"
            )

        try:
            # Start database transaction
            db_document = BlockchainDocument(
                document_hash=blockchain_entry["document_hash"].hex(),
                original_filename=file.filename,
                file_path=file_path,
                department=department,  # Use the required department parameter
                classification=classification,
                created_by=user.id,
                document_metadata={
                    "blockchain_tx": blockchain_entry["transaction_hash"],
                    "content_hash": blockchain_entry["content_hash"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            db.add(db_document)
            
            # Flush to get the document ID without committing
            db.flush()
            
            if not db_document.id:
                raise ValueError("Failed to generate document ID")

            # Now create the audit entry with the document ID
            audit_entry = BlockchainAuditEntry(
                document_id=db_document.id,
                action_type=BlockchainActionType.CREATED,
                user_id=user.id,
                transaction_hash=blockchain_entry["transaction_hash"],
                action_metadata={
                    "initial_version": "1.0",
                    "document_hash": blockchain_entry["document_hash"].hex(),
                    "content_hash": blockchain_entry["content_hash"],
                    "department": department  # Add department to audit metadata
                }
            )
            db.add(audit_entry)
            
            # Now commit both records
            db.commit()
            db.refresh(db_document)
            
            logger.info(f"Successfully created blockchain document {db_document.id}")
            return db_document

        except Exception as db_error:
            logger.error(f"Database operation failed: {db_error}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save document records: {str(db_error)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_blockchain_document: {str(e)}")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Clean up file if something went wrong
        if file_path and os.path.exists(file_path) and 'db_document' not in locals():
            os.remove(file_path)

@router.put("/documents/{document_id}", response_model=BlockchainDocumentResponse)
async def update_blockchain_document(
    document_id: UUID,
    user_data: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    modification_type: str = Query(..., description="Type of modification made"),
    previous_version: str = Query(..., description="Previous version number"),
    change_description: str = Query(..., description="Description of changes made")
):
    """Update a blockchain-tracked document"""
    try:
        # Get user from database
        user = db.query(User).filter(User.email == user_data["email"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check document exists
        document = db.query(BlockchainDocument).filter_by(id=document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Save updated file
        with open(document.file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get user's blockchain key
        blockchain_key = db.query(BlockchainUserKeys).filter_by(user_id=user.id).first()
        if not blockchain_key:
            raise HTTPException(status_code=400, detail="User has no blockchain key")

        # Get Solana keypair using the utility function
        user_keypair = get_user_keypair(db, str(user.id))
        if not user_keypair:
            raise HTTPException(status_code=500, detail="Failed to decrypt user's blockchain key")

        # Track update in blockchain
        blockchain_entry = await blockchain_service.track_document_update(
            file_path=document.file_path,
            user_keypair=user_keypair,
            modification_type=modification_type,
            previous_version=previous_version,
            change_description=change_description,
            user_id=str(user.id),  # Convert UUID to string for blockchain
            username=user.full_name
        )

        if not blockchain_entry:
            raise HTTPException(status_code=500, detail="Failed to update blockchain entry")

        # Update document record
        document.document_hash = blockchain_entry["document_hash"].hex()
        document.updated_at = datetime.now()
        document.document_metadata.update({
            "last_blockchain_tx": blockchain_entry["transaction_hash"],
            "last_content_hash": blockchain_entry["content_hash"]
        })

        # Create audit entry
        audit_entry = BlockchainAuditEntry(
            document_id=document.id,
            action_type=BlockchainActionType.MODIFIED,
            user_id=user.id,  # UUID type
            transaction_hash=blockchain_entry["transaction_hash"],
            previous_version=previous_version,
            action_metadata={
                "modification_type": modification_type,
                "change_description": change_description
            }
        )
        db.add(audit_entry)
        
        db.commit()
        db.refresh(document)
        
        return document

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{document_id}/access", response_model=BlockchainAuditEntryResponse)
async def track_document_access(
    document_id: UUID,
    user_data: Dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    access_type: str = Query(..., description="Type of access (READ, WRITE, etc.)"),
    purpose: str = Query(..., description="Purpose of accessing the document")
):
    """Track document access in blockchain"""
    try:
        # Get user from database
        user = db.query(User).filter(User.email == user_data["email"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check document exists
        document = db.query(BlockchainDocument).filter_by(id=document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get user's blockchain key
        blockchain_key = db.query(BlockchainUserKeys).filter_by(user_id=user.id).first()
        if not blockchain_key:
            raise HTTPException(status_code=400, detail="User has no blockchain key")

        # Get Solana keypair using the utility function
        user_keypair = get_user_keypair(db, str(user.id))
        if not user_keypair:
            raise HTTPException(status_code=500, detail="Failed to decrypt user's blockchain key")

        # Track access in blockchain
        blockchain_entry = await blockchain_service.track_document_access(
            file_path=document.file_path,
            user_keypair=user_keypair,
            access_type=access_type,
            purpose=purpose,
            user_id=str(user.id),  # Convert UUID to string for blockchain
            username=user.full_name
        )

        if not blockchain_entry:
            raise HTTPException(status_code=500, detail="Failed to track document access")

        # Create audit entry
        audit_entry = BlockchainAuditEntry(
            document_id=document.id,
            action_type=BlockchainActionType.ACCESSED,
            user_id=user.id,  # UUID type
            transaction_hash=blockchain_entry["transaction_hash"],
            action_metadata={
                "access_type": access_type,
                "purpose": purpose
            }
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        
        return audit_entry

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}/history", response_model=BlockchainDocumentHistory)
async def get_document_history(
    document_id: UUID,
    user_data: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get complete document history from blockchain"""
    try:
        # Get user from database
        user = db.query(User).filter(User.email == user_data["email"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check document exists
        document = db.query(BlockchainDocument).filter_by(id=document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get blockchain history
        blockchain_history = await blockchain_service.get_document_history(document.file_path)
        if blockchain_history is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve blockchain history")

        # Get audit entries
        audit_entries = db.query(BlockchainAuditEntry).filter_by(document_id=document_id).all()

        return BlockchainDocumentHistory(
            document=document,
            audit_entries=audit_entries
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}/verify")
async def verify_document_integrity(
    document_id: UUID,
    user_data: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify document integrity using blockchain hash"""
    try:
        # Get user from database
        user = db.query(User).filter(User.email == user_data["email"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check document exists
        document = db.query(BlockchainDocument).filter_by(id=document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Verify integrity
        is_valid = await blockchain_service.verify_document_integrity(
            document.file_path,
            document.document_hash
        )

        return {
            "document_id": document_id,
            "is_valid": is_valid,
            "blockchain_hash": document.document_hash
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 