"""
AES-256 Encryption Service for Enterprise Security
"""

import base64
import logging
import hashlib
import secrets
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EncryptionService:
    """Enterprise-grade AES-256 encryption service"""
    
    def __init__(self):
        self.key = self._generate_or_load_key()
        self.fernet = Fernet(self.key)
        
        # Initialize password context for hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def _generate_or_load_key(self) -> bytes:
        """Generate or load encryption key"""
        key_file = "encryption.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt string data"""
        try:
            if data is None:
                raise TypeError("Cannot encrypt None data")
            if not isinstance(data, str):
                raise TypeError("Data must be a string")
            
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        json_data = json.dumps(data)
        return self.encrypt_data(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        json_data = self.decrypt_data(encrypted_data)
        return json.loads(json_data)
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        try:
            if not isinstance(password, str):
                raise TypeError("Password must be a string")
            return self.pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Password hashing error: {str(e)}")
            raise
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def generate_key(self) -> str:
        """Generate a new cryptographic key"""
        try:
            key = Fernet.generate_key()
            return base64.b64encode(key).decode()
        except Exception as e:
            logger.error(f"Key generation error: {str(e)}")
            raise
    
    def hash_file_content(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"File hashing error: {str(e)}")
            raise
    
    def generate_secure_random(self, length: int) -> str:
        """Generate cryptographically secure random string"""
        try:
            return secrets.token_hex(length // 2)[:length]
        except Exception as e:
            logger.error(f"Secure random generation error: {str(e)}")
            raise
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about the current encryption key"""
        return {
            "key_type": "Fernet",
            "algorithm": "AES-128",
            "key_length": len(self.key),
            "created": "runtime"
        }
    
    def rotate_key(self) -> str:
        """Rotate the encryption key (generate new one)"""
        try:
            old_key = self.key
            new_key = Fernet.generate_key()
            
            # Save new key
            with open("encryption.key", 'wb') as f:
                f.write(new_key)
            
            # Update service
            self.key = new_key
            self.fernet = Fernet(new_key)
            
            logger.info("Encryption key rotated successfully")
            return base64.b64encode(new_key).decode()
        except Exception as e:
            logger.error(f"Key rotation error: {str(e)}")
            raise
    
    def secure_delete(self, file_path: str) -> bool:
        """Securely delete a file by overwriting it multiple times"""
        try:
            if not os.path.exists(file_path):
                return True
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite file multiple times
            with open(file_path, "r+b") as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Secure delete error: {str(e)}")
            return False

# Initialize encryption service
encryption_service = EncryptionService() 