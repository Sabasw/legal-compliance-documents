from cryptography.fernet import Fernet
from solders.keypair import Keypair
import base64
import os
import hashlib
import logging
import traceback
from datetime import datetime
from app.database.models.models import User
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Initialize encryption key
def get_encryption_key():
    """Get or create the encryption key"""
    key_path = "blockchain_key.key"
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return Fernet(f.read())
    else:
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
        return Fernet(key)

ENCRYPTION_KEY = get_encryption_key()

def generate_user_keypair() -> tuple:
    """Generate a new Solana keypair"""
    try:
        # Generate new keypair
        keypair = Keypair()
        
        # Get the full keypair bytes (64 bytes)
        keypair_bytes = keypair.to_bytes()
        
        # Encode and encrypt
        keypair_b64 = base64.b64encode(keypair_bytes).decode()
        encrypted_keypair = ENCRYPTION_KEY.encrypt(keypair_b64.encode()).decode()
        
        # Test decryption
        test_decrypt = ENCRYPTION_KEY.decrypt(encrypted_keypair.encode()).decode()
        test_bytes = base64.b64decode(test_decrypt)
        
        if len(test_bytes) != 64:
            raise ValueError(f"Invalid keypair length after decryption: {len(test_bytes)}")
            
        # Create test keypair from full bytes
        test_keypair = Keypair.from_bytes(test_bytes)
        
        # Verify the public key matches
        if str(test_keypair.pubkey()) != str(keypair.pubkey()):
            raise ValueError("Keypair verification failed")
            
        return str(keypair.pubkey()), encrypted_keypair
    except Exception as e:
        logger.error(f"Failed to generate keypair: {str(e)}\n{traceback.format_exc()}")
        raise

def ensure_user_blockchain_keys(db: Session, user_id: str) -> dict:
    """Ensure user has valid blockchain keys, regenerate if invalid"""
    try:
        from app.database.models.blockchain_models import BlockchainUserKeys
        
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            logger.error(f"User {user_id} not found")
            return {"success": False, "message": "User not found"}

        # Check if user has blockchain keys in the BlockchainUserKeys table
        blockchain_keys = db.query(BlockchainUserKeys).filter_by(user_id=user_id).first()
        
        if not blockchain_keys:
            logger.info(f"No blockchain keys found for user {user_id}, generating new one")
            public_key, encrypted_keypair = generate_user_keypair()
            
            # Create new blockchain keys record
            new_blockchain_keys = BlockchainUserKeys(
                user_id=user_id,
                public_key=public_key,
                encrypted_private_key=encrypted_keypair
            )
            db.add(new_blockchain_keys)
            db.commit()
            return {"success": True, "message": "Generated new blockchain keys"}

        # Verify existing key can be decrypted
        try:
            encrypted_bytes = blockchain_keys.encrypted_private_key.encode()
            keypair_b64 = ENCRYPTION_KEY.decrypt(encrypted_bytes).decode()
            keypair_bytes = base64.b64decode(keypair_b64)
            
            # Test keypair creation
            test_keypair = Keypair.from_bytes(keypair_bytes)
            logger.debug(f"Successfully validated keypair with public key: {test_keypair.pubkey()}")
            return {"success": True, "message": "Existing blockchain keys are valid"}
            
        except Exception as e:
            logger.warning(f"Invalid blockchain key for user {user_id}, regenerating: {str(e)}")
            public_key, encrypted_keypair = generate_user_keypair()
            
            # Update existing record
            blockchain_keys.public_key = public_key
            blockchain_keys.encrypted_private_key = encrypted_keypair
            db.commit()
            return {"success": True, "message": "Regenerated invalid blockchain keys"}

    except Exception as e:
        logger.error(f"Failed to ensure blockchain keys: {str(e)}\n{traceback.format_exc()}")
        return {"success": False, "message": f"Error ensuring blockchain keys: {str(e)}"}

def get_user_keypair(db: Session, user_id: str) -> Keypair:
    """Get user's Solana keypair"""
    try:
        from app.database.models.blockchain_models import BlockchainUserKeys
        
        # First ensure user has valid blockchain keys
        status = ensure_user_blockchain_keys(db, user_id)
        if not status["success"]:
            logger.error(f"Failed to ensure blockchain keys: {status['message']}")
            return None

        # Get blockchain keys from the BlockchainUserKeys table
        blockchain_keys = db.query(BlockchainUserKeys).filter_by(user_id=user_id).first()
        if not blockchain_keys:
            logger.error(f"No blockchain keys found for user {user_id}")
            return None

        try:
            # Decrypt keypair
            encrypted_bytes = blockchain_keys.encrypted_private_key.encode()
            keypair_b64 = ENCRYPTION_KEY.decrypt(encrypted_bytes).decode()
            keypair_bytes = base64.b64decode(keypair_b64)
            
            if len(keypair_bytes) != 64:
                raise ValueError(f"Invalid keypair length: {len(keypair_bytes)}")
            
            # Create keypair from full bytes
            keypair = Keypair.from_bytes(keypair_bytes)
            logger.debug(f"Successfully created keypair with public key: {keypair.pubkey()}")
            return keypair

        except Exception as decrypt_error:
            logger.error(f"Failed to decrypt blockchain key: {decrypt_error}\n{traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"Failed to get user keypair: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_content_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file content"""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate content hash: {str(e)}")
        return None 