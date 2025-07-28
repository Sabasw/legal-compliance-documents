"""
Blockchain Audit Trail Configuration
Configuration settings for the immutable audit trail system
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class BlockchainConfig:
    """Configuration class for blockchain audit trail system"""
    
    # Blockchain Network Configuration
    BLOCKCHAIN_RPC_URL = os.getenv('BLOCKCHAIN_RPC_URL', 'http://localhost:8545')
    BLOCKCHAIN_CHAIN_ID = int(os.getenv('BLOCKCHAIN_CHAIN_ID', '1337'))
    BLOCKCHAIN_PRIVATE_KEY = os.getenv('BLOCKCHAIN_PRIVATE_KEY')
    AUDIT_CONTRACT_ADDRESS = os.getenv('AUDIT_CONTRACT_ADDRESS')
    
    # Proof of Work Configuration
    POW_DIFFICULTY = int(os.getenv('POW_DIFFICULTY', '4'))  # Number of leading zeros required
    POW_TIMEOUT = int(os.getenv('POW_TIMEOUT', '30'))  # Timeout in seconds for PoW calculation
    
    # Audit Trail Configuration
    CRITICAL_ACTIONS = {
        "document": [
            "upload", "delete", "modify", "compliance_check", 
            "risk_assessment", "sign", "verify"
        ],
        "compliance_rule": [
            "create", "update", "delete", "activate", "deactivate"
        ],
        "user": [
            "create", "delete", "role_change", "permission_change"
        ],
        "audit": [
            "export", "delete", "modify", "verify"
        ],
        "legal_document": [
            "create", "modify", "delete", "sign", "verify", "notarize"
        ],
        "contract": [
            "create", "modify", "delete", "sign", "execute", "terminate"
        ],
        "evidence": [
            "upload", "delete", "modify", "verify", "chain_of_custody"
        ],
        "compliance_report": [
            "generate", "submit", "approve", "reject"
        ]
    }
    
    # Blockchain Gas Configuration
    GAS_LIMIT = int(os.getenv('BLOCKCHAIN_GAS_LIMIT', '200000'))
    GAS_PRICE_STRATEGY = os.getenv('BLOCKCHAIN_GAS_PRICE_STRATEGY', 'auto')  # auto, manual, fast
    
    # Audit Record Configuration
    MAX_RECORD_SIZE = int(os.getenv('MAX_RECORD_SIZE', '1024'))  # Maximum record size in bytes
    RECORD_RETENTION_DAYS = int(os.getenv('RECORD_RETENTION_DAYS', '2555'))  # 7 years
    
    # Verification Configuration
    VERIFICATION_INTERVAL = int(os.getenv('VERIFICATION_INTERVAL', '3600'))  # Verify chain every hour
    VERIFICATION_TIMEOUT = int(os.getenv('VERIFICATION_TIMEOUT', '300'))  # 5 minutes timeout
    
    # Export Configuration
    EXPORT_FORMATS = ['json', 'csv', 'pdf']
    EXPORT_MAX_RECORDS = int(os.getenv('EXPORT_MAX_RECORDS', '10000'))
    
    # Security Configuration
    HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')
    SIGNATURE_ALGORITHM = os.getenv('SIGNATURE_ALGORITHM', 'ecdsa')
    
    # Compliance Configuration
    COMPLIANCE_STANDARDS = {
        "SOX": {
            "retention_period": 7,  # years
            "audit_requirements": ["immutable_logs", "chain_of_custody", "access_controls"],
            "verification_frequency": "daily"
        },
        "GDPR": {
            "retention_period": 6,  # years
            "audit_requirements": ["data_processing_logs", "consent_records", "breach_notifications"],
            "verification_frequency": "weekly"
        },
        "HIPAA": {
            "retention_period": 6,  # years
            "audit_requirements": ["access_logs", "disclosure_records", "security_incidents"],
            "verification_frequency": "monthly"
        },
        "ISO27001": {
            "retention_period": 3,  # years
            "audit_requirements": ["security_events", "access_controls", "change_management"],
            "verification_frequency": "quarterly"
        }
    }
    
    @classmethod
    def get_critical_actions(cls) -> Dict[str, list]:
        """Get critical actions configuration"""
        return cls.CRITICAL_ACTIONS
    
    @classmethod
    def is_critical_action(cls, action: str, entity_type: str) -> bool:
        """Check if an action is critical and requires blockchain verification"""
        return (
            entity_type in cls.CRITICAL_ACTIONS and 
            action in cls.CRITICAL_ACTIONS[entity_type]
        )
    
    @classmethod
    def get_compliance_requirements(cls, standard: str) -> Dict[str, Any]:
        """Get compliance requirements for a specific standard"""
        return cls.COMPLIANCE_STANDARDS.get(standard, {})
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate the blockchain configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required environment variables
        if not cls.BLOCKCHAIN_PRIVATE_KEY:
            validation_results["warnings"].append("BLOCKCHAIN_PRIVATE_KEY not set - using in-memory storage only")
        
        if not cls.AUDIT_CONTRACT_ADDRESS:
            validation_results["warnings"].append("AUDIT_CONTRACT_ADDRESS not set - blockchain verification disabled")
        
        # Check network connectivity
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(cls.BLOCKCHAIN_RPC_URL))
            if not w3.is_connected():
                validation_results["errors"].append(f"Cannot connect to blockchain network: {cls.BLOCKCHAIN_RPC_URL}")
                validation_results["valid"] = False
        except Exception as e:
            validation_results["errors"].append(f"Blockchain connection error: {str(e)}")
            validation_results["valid"] = False
        
        # Check configuration values
        if cls.POW_DIFFICULTY < 1 or cls.POW_DIFFICULTY > 8:
            validation_results["warnings"].append("POW_DIFFICULTY should be between 1 and 8")
        
        if cls.GAS_LIMIT < 100000 or cls.GAS_LIMIT > 1000000:
            validation_results["warnings"].append("GAS_LIMIT should be between 100,000 and 1,000,000")
        
        return validation_results
    
    @classmethod
    def get_deployment_info(cls) -> Dict[str, Any]:
        """Get deployment information"""
        return {
            "blockchain_network": {
                "rpc_url": cls.BLOCKCHAIN_RPC_URL,
                "chain_id": cls.BLOCKCHAIN_CHAIN_ID,
                "contract_address": cls.AUDIT_CONTRACT_ADDRESS
            },
            "configuration": {
                "pow_difficulty": cls.POW_DIFFICULTY,
                "gas_limit": cls.GAS_LIMIT,
                "max_record_size": cls.MAX_RECORD_SIZE,
                "retention_days": cls.RECORD_RETENTION_DAYS
            },
            "compliance_standards": list(cls.COMPLIANCE_STANDARDS.keys()),
            "critical_entities": list(cls.CRITICAL_ACTIONS.keys())
        }

# Global configuration instance
blockchain_config = BlockchainConfig() 