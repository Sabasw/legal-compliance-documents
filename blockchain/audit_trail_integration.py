import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from solders.keypair import Keypair
from blockchain.audit_trail_manager import DocumentAuditTrail, AuditTrailConfig, DocumentAction
import logging
from datetime import datetime

class AuditTrailIntegration:
    """Helper class for integrating audit trail functionality into existing projects"""
    
    def __init__(self, config_path: str = "audit_trail_config.json"):
        """Initialize with configuration file"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.audit_trail = self._initialize_audit_trail()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for audit trail operations"""
        logger = logging.getLogger("AuditTrail")
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('audit_trail.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def _initialize_audit_trail(self) -> DocumentAuditTrail:
        """Initialize the audit trail manager with configuration"""
        config = AuditTrailConfig(
            program_id=self.config['solana']['program_id'],
            rpc_endpoint=self.config['solana']['rpc_endpoint'],
            retention_period=self.config['audit_trail']['retention_period'],
            jurisdiction=self.config['audit_trail']['jurisdiction'],
            default_access_level=self.config['audit_trail']['default_access_level']
        )
        return DocumentAuditTrail(config)
    
    def _validate_document(self, file_path: Union[str, Path], content: bytes) -> bool:
        """Validate document before processing"""
        file_path = Path(file_path)
        
        # Check file type
        if file_path.suffix.upper().replace('.', '') not in self.config['audit_trail']['allowed_document_types']:
            self.logger.error(f"Unsupported file type: {file_path.suffix}")
            return False
            
        # Check file size
        if len(content) > self.config['audit_trail']['max_file_size_bytes']:
            self.logger.error(f"File too large: {len(content)} bytes")
            return False
            
        return True
    
    def track_new_document(self, 
                          file_path: Union[str, Path],
                          user_keypair: Keypair,
                          department: str,
                          classification: str = "Confidential",
                          **kwargs) -> Optional[Dict[str, Any]]:
        """
        Track a new document in the audit trail system
        
        Args:
            file_path: Path to the document
            user_keypair: User's Solana keypair
            department: Department the document belongs to
            classification: Document classification level
            **kwargs: Additional metadata
        
        Returns:
            Dict containing the audit entry or None if failed
        """
        try:
            file_path = Path(file_path)
            
            # Read document content
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # Validate document
            if not self._validate_document(file_path, content):
                return None
                
            # Track document creation
            entry = self.audit_trail.track_document_creation(
                document_content=content,
                user_pubkey=user_keypair.pubkey(),
                document_title=file_path.name,
                document_type=file_path.suffix.upper().replace('.', ''),
                department=department,
                classification=classification,
                **kwargs
            )
            
            self.logger.info(f"Successfully tracked new document: {file_path.name}")
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to track document: {e}")
            return None
    
    def track_document_update(self,
                            file_path: Union[str, Path],
                            user_keypair: Keypair,
                            modification_type: str,
                            previous_version: str,
                            change_description: str,
                            **kwargs) -> Optional[Dict[str, Any]]:
        """Track document updates"""
        try:
            file_path = Path(file_path)
            
            # Read document content
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # Validate document
            if not self._validate_document(file_path, content):
                return None
                
            # Track modification
            entry = self.audit_trail.track_document_modification(
                document_content=content,
                user_pubkey=user_keypair.pubkey(),
                modification_type=modification_type,
                document_title=file_path.name,
                previous_version=previous_version,
                change_description=change_description,
                **kwargs
            )
            
            self.logger.info(f"Successfully tracked document update: {file_path.name}")
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to track document update: {e}")
            return None
    
    def track_document_access(self,
                            file_path: Union[str, Path],
                            user_keypair: Keypair,
                            access_type: str,
                            purpose: str,
                            **kwargs) -> Optional[Dict[str, Any]]:
        """Track document access"""
        try:
            file_path = Path(file_path)
            
            # Read document content
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # Track access
            entry = self.audit_trail.track_document_access(
                document_content=content,
                user_pubkey=user_keypair.pubkey(),
                access_type=access_type,
                purpose=purpose,
                **kwargs
            )
            
            self.logger.info(f"Successfully tracked document access: {file_path.name}")
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to track document access: {e}")
            return None
    
    def get_document_audit_history(self, file_path: Union[str, Path]) -> Optional[list]:
        """Get complete audit history for a document"""
        try:
            file_path = Path(file_path)
            
            # Read document content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Calculate document hash
            doc_hash = self.audit_trail.audit_manager.calculate_document_hash(content).hex()
            
            # Get history
            history = self.audit_trail.get_document_history(doc_hash)
            
            self.logger.info(f"Retrieved audit history for: {file_path.name}")
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get document history: {e}")
            return None
    
    def verify_document(self, file_path: Union[str, Path], expected_hash: str) -> bool:
        """Verify document integrity"""
        try:
            file_path = Path(file_path)
            
            # Read document content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Verify integrity
            is_valid = self.audit_trail.verify_document_integrity(content, expected_hash)
            
            self.logger.info(f"Document verification result for {file_path.name}: {'Valid' if is_valid else 'Invalid'}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Failed to verify document: {e}")
            return False

# Example usage
def example_integration():
    # Initialize integration
    audit_integration = AuditTrailIntegration()
    
    # Create test user
    user_keypair = Keypair()
    
    # Track new document
    entry = audit_integration.track_new_document(
        file_path="7.pdf",
        user_keypair=user_keypair,
        department="Legal",
        classification="Confidential",
        author="John Doe",
        project="Contract Review"
    )
    
    if entry:
        # Get document hash
        doc_hash = entry['document_hash'].hex()
        
        # Track document access
        audit_integration.track_document_access(
            file_path="7.pdf",
            user_keypair=user_keypair,
            access_type="READ",
            purpose="Legal Review"
        )
        
        # Get document history
        history = audit_integration.get_document_audit_history("7.pdf")
        
        # Verify document
        is_valid = audit_integration.verify_document("7.pdf", doc_hash)
        
        return entry, history, is_valid
    
    return None, None, False

if __name__ == "__main__":
    # Run example
    entry, history, is_valid = example_integration()
    if entry:
        print(f"\n✅ Integration test successful!")
        print(f"📄 Document Hash: {entry['document_hash'].hex()}")
        print(f"📜 History Entries: {len(history) if history else 0}")
        print(f"🔐 Document Integrity: {'Valid' if is_valid else 'Invalid'}")
    else:
        print("\n❌ Integration test failed!") 