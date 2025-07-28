from blockchain.blockchain_data import AuditTrailManager
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64
from dataclasses import dataclass
from enum import Enum

class DocumentAction(Enum):
    """Enum for document actions that can be tracked"""
    CREATED = "DOCUMENT_CREATED"
    MODIFIED = "DOCUMENT_MODIFIED"
    ACCESSED = "DOCUMENT_ACCESSED"
    REVIEWED = "DOCUMENT_REVIEWED"
    DELETED = "DOCUMENT_DELETED"
    SHARED = "DOCUMENT_SHARED"
    PRINTED = "DOCUMENT_PRINTED"
    SIGNED = "DOCUMENT_SIGNED"

@dataclass
class AuditTrailConfig:
    """Configuration for audit trail management"""
    program_id: str
    rpc_endpoint: str = "https://api.testnet.solana.com"
    retention_period: int = 31536000  # 1 year in seconds
    jurisdiction: str = "US"
    default_access_level: int = 1

class DocumentAuditTrail:
    """Main class for managing document audit trails"""
    
    def __init__(self, config: AuditTrailConfig):
        """Initialize with configuration"""
        self.config = config
        self.client = Client(config.rpc_endpoint)
        self.program_id = Pubkey.from_string(config.program_id)
        self.audit_manager = AuditTrailManager(self.client, self.program_id)
        
    def track_document_action(self,
                            action: DocumentAction,
                            document_content: bytes,
                            user_pubkey: Pubkey,
                            metadata: Dict[str, Any] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Track a document action in the audit trail
        
        Args:
            action: The action being performed on the document
            document_content: The document's content
            user_pubkey: The public key of the user performing the action
            metadata: Additional metadata about the action
            **kwargs: Additional keyword arguments for flexibility
            
        Returns:
            Dict containing the audit entry details
        """
        metadata = metadata or {}
        metadata.update(kwargs)
        
        # Add basic file information to metadata
        metadata.update({
            'file_size': len(document_content),
            'timestamp': datetime.now().isoformat(),
            'action_type': action.value
        })
        
        # Create the audit entry
        entry = self.audit_manager.create_audit_entry(
            action=action.value,
            user_pubkey=user_pubkey,
            document_content=document_content,
            record_type="DOCUMENT",
            jurisdiction=self.config.jurisdiction,
            retention_period=self.config.retention_period,
            access_level=self.config.default_access_level,
            metadata=metadata
        )
        
        return entry
    
    def get_document_history(self, document_hash: str) -> List[Dict[str, Any]]:
        """
        Get the complete history of a document
        
        Args:
            document_hash: The hash of the document to look up
            
        Returns:
            List of audit entries for the document
        """
        entries = self.audit_manager.get_audit_trail({
            "document_hash": bytes.fromhex(document_hash)
        })
        
        return entries
    
    def verify_document_integrity(self, document_content: bytes, document_hash: str) -> bool:
        """
        Verify if a document matches its recorded hash
        
        Args:
            document_content: The current content of the document
            document_hash: The hash to verify against
            
        Returns:
            bool: True if document is unchanged, False otherwise
        """
        current_hash = self.audit_manager.calculate_document_hash(document_content)
        return current_hash.hex() == document_hash
    
    def track_document_creation(self,
                              document_content: bytes,
                              user_pubkey: Pubkey,
                              document_title: str,
                              document_type: str = "PDF",
                              **kwargs) -> Dict[str, Any]:
        """
        Track the creation of a new document
        """
        metadata = {
            'document_title': document_title,
            'document_type': document_type,
            'document_version': '1.0',
            'created_by': str(user_pubkey),
            **kwargs
        }
        
        return self.track_document_action(
            action=DocumentAction.CREATED,
            document_content=document_content,
            user_pubkey=user_pubkey,
            metadata=metadata
        )
    
    def track_document_modification(self,
                                  document_content: bytes,
                                  user_pubkey: Pubkey,
                                  modification_type: str,
                                  document_title: str,
                                  previous_version: str,
                                  **kwargs) -> Dict[str, Any]:
        """
        Track a modification to an existing document
        """
        metadata = {
            'document_title': document_title,
            'modified_by': str(user_pubkey),
            'modification_type': modification_type,
            'previous_version': previous_version,
            'new_version': f"{float(previous_version) + 0.1:.1f}",
            **kwargs
        }
        
        return self.track_document_action(
            action=DocumentAction.MODIFIED,
            document_content=document_content,
            user_pubkey=user_pubkey,
            metadata=metadata
        )
    
    def track_document_access(self,
                            document_content: bytes,
                            user_pubkey: Pubkey,
                            access_type: str,
                            purpose: str,
                            **kwargs) -> Dict[str, Any]:
        """
        Track document access events
        """
        metadata = {
            'accessor': str(user_pubkey),
            'access_type': access_type,
            'access_purpose': purpose,
            **kwargs
        }
        
        return self.track_document_action(
            action=DocumentAction.ACCESSED,
            document_content=document_content,
            user_pubkey=user_pubkey,
            metadata=metadata
        )

# Example usage:
def example_usage():
    # Configuration
    config = AuditTrailConfig(
        program_id="EJmeW47zDebNgLZT3oiHmH5zBkW4yLb1d2MVjQZHsQ8G",
        rpc_endpoint="https://api.testnet.solana.com",
        retention_period=31536000,  # 1 year
        jurisdiction="US",
        default_access_level=1
    )
    
    # Initialize audit trail manager
    audit_trail = DocumentAuditTrail(config)
    
    # Example: Track document creation
    with open("7.pdf", 'rb') as f:
        document_content = f.read()
    
    user_keypair = Keypair()  # In real usage, this would be the actual user's keypair
    
    # Track document creation
    creation_entry = audit_trail.track_document_creation(
        document_content=document_content,
        user_pubkey=user_keypair.pubkey(),
        document_title="7.pdf",
        document_type="PDF",
        department="Legal",
        classification="Confidential"
    )
    
    # Get document history
    doc_hash = creation_entry['document_hash'].hex()
    history = audit_trail.get_document_history(doc_hash)
    
    # Verify document integrity
    is_valid = audit_trail.verify_document_integrity(document_content, doc_hash)
    
    return creation_entry, history, is_valid

if __name__ == "__main__":
    # Run example
    entry, history, is_valid = example_usage()
    print(f"\n✅ Document tracked successfully!")
    print(f"📄 Document Hash: {entry['document_hash'].hex()}")
    print(f"📜 History Entries: {len(history)}")
    print(f"🔐 Document Integrity: {'Valid' if is_valid else 'Invalid'}") 