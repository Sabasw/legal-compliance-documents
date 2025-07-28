import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from web3 import Web3
from web3.middleware import geth_poa_middleware
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class AuditRecord:
    """Immutable audit record structure"""
    record_id: str
    timestamp: str
    user_id: str
    action: str
    entity_type: str
    entity_id: str
    document_id: Optional[str]
    changes: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    data_hash: str
    previous_hash: Optional[str]
    nonce: int
    signature: Optional[str] = None

class BlockchainAuditService:
    """
    Blockchain-based audit trail service for immutable legal records
    """
    
    def __init__(self):
        # Initialize Web3 connection (using local blockchain or testnet)
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('BLOCKCHAIN_RPC_URL', 'http://localhost:8545')))
        
        # Add POA middleware for test networks
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Contract ABI for audit trail smart contract
        self.contract_abi = self._get_contract_abi()
        self.contract_address = os.getenv('AUDIT_CONTRACT_ADDRESS')
        
        # Initialize contract if address is provided
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
        
        # Private key for signing transactions (should be stored securely)
        self.private_key = os.getenv('BLOCKCHAIN_PRIVATE_KEY')
        
        # Chain ID for the blockchain network
        self.chain_id = int(os.getenv('BLOCKCHAIN_CHAIN_ID', '1337'))
        
        # In-memory audit chain (for demo/testing purposes)
        self.audit_chain: List[AuditRecord] = []
        self.last_hash: Optional[str] = None
        
    def _get_contract_abi(self) -> List[Dict]:
        """Get the smart contract ABI for audit trail"""
        return [
            {
                "inputs": [
                    {"internalType": "string", "name": "recordId", "type": "string"},
                    {"internalType": "string", "name": "dataHash", "type": "string"},
                    {"internalType": "string", "name": "previousHash", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "string", "name": "metadata", "type": "string"}
                ],
                "name": "addAuditRecord",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "recordId", "type": "string"}
                ],
                "name": "getAuditRecord",
                "outputs": [
                    {"internalType": "string", "name": "dataHash", "type": "string"},
                    {"internalType": "string", "name": "previousHash", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "string", "name": "metadata", "type": "string"},
                    {"internalType": "bool", "name": "exists", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getChainLength",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _calculate_record_hash(self, record: AuditRecord) -> str:
        """Calculate hash of audit record"""
        record_data = {
            'record_id': record.record_id,
            'timestamp': record.timestamp,
            'user_id': record.user_id,
            'action': record.action,
            'entity_type': record.entity_type,
            'entity_id': record.entity_id,
            'document_id': record.document_id,
            'changes': record.changes,
            'ip_address': record.ip_address,
            'user_agent': record.user_agent,
            'previous_hash': record.previous_hash,
            'nonce': record.nonce
        }
        return self._calculate_hash(json.dumps(record_data, sort_keys=True))
    
    def _find_nonce(self, record: AuditRecord, difficulty: int = 4) -> int:
        """Simple proof-of-work to find nonce"""
        target = '0' * difficulty
        nonce = 0
        
        while True:
            record.nonce = nonce
            hash_result = self._calculate_record_hash(record)
            if hash_result.startswith(target):
                return nonce
            nonce += 1
    
    def create_audit_record(
        self,
        user_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        document_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditRecord:
        """
        Create an immutable audit record
        """
        try:
            # Generate unique record ID
            record_id = f"audit_{int(time.time())}_{hash(f'{user_id}{action}{entity_id}')}"
            
            # Create audit record
            record = AuditRecord(
                record_id=record_id,
                timestamp=datetime.utcnow().isoformat(),
                user_id=user_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                document_id=document_id,
                changes=changes or {},
                ip_address=ip_address,
                user_agent=user_agent,
                data_hash="",  # Will be calculated after nonce
                previous_hash=self.last_hash,
                nonce=0
            )
            
            # Calculate data hash
            record.data_hash = self._calculate_record_hash(record)
            
            # Find proof-of-work nonce
            record.nonce = self._find_nonce(record)
            
            # Update data hash with final nonce
            record.data_hash = self._calculate_record_hash(record)
            
            # Update last hash
            self.last_hash = record.data_hash
            
            logger.info(f"Created audit record: {record_id}")
            return record
            
        except Exception as e:
            logger.error(f"Error creating audit record: {str(e)}")
            raise
    
    def add_to_blockchain(self, record: AuditRecord) -> Optional[str]:
        """
        Add audit record to blockchain
        """
        try:
            if not self.contract or not self.private_key:
                logger.warning("Blockchain not configured, storing in memory only")
                self.audit_chain.append(record)
                return None
            
            # Prepare transaction
            metadata = json.dumps({
                'user_id': record.user_id,
                'action': record.action,
                'entity_type': record.entity_type,
                'entity_id': record.entity_id,
                'document_id': record.document_id,
                'changes': record.changes,
                'ip_address': record.ip_address,
                'user_agent': record.user_agent
            })
            
            # Build transaction
            transaction = self.contract.functions.addAuditRecord(
                record.record_id,
                record.data_hash,
                record.previous_hash or "",
                int(time.time()),
                metadata
            ).build_transaction({
                'from': self.w3.eth.account.from_key(self.private_key).address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(
                    self.w3.eth.account.from_key(self.private_key).address
                ),
                'chainId': self.chain_id
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Audit record added to blockchain: {record.record_id}, TX: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error adding to blockchain: {str(e)}")
            # Fallback to in-memory storage
            self.audit_chain.append(record)
            return None
    
    def verify_audit_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain
        """
        try:
            if not self.audit_chain:
                return {"valid": True, "message": "Empty audit chain"}
            
            verification_results = []
            previous_hash = None
            
            for i, record in enumerate(self.audit_chain):
                # Verify hash calculation
                expected_hash = self._calculate_record_hash(record)
                hash_valid = expected_hash == record.data_hash
                
                # Verify previous hash link
                link_valid = record.previous_hash == previous_hash
                
                # Verify proof-of-work
                pow_valid = record.data_hash.startswith('0000')  # 4 leading zeros
                
                verification_results.append({
                    'record_id': record.record_id,
                    'index': i,
                    'hash_valid': hash_valid,
                    'link_valid': link_valid,
                    'pow_valid': pow_valid,
                    'timestamp': record.timestamp
                })
                
                previous_hash = record.data_hash
            
            # Check if any records are invalid
            invalid_records = [r for r in verification_results if not all([r['hash_valid'], r['link_valid'], r['pow_valid']])]
            
            return {
                "valid": len(invalid_records) == 0,
                "total_records": len(self.audit_chain),
                "invalid_records": len(invalid_records),
                "verification_results": verification_results,
                "message": "Audit chain verification completed"
            }
            
        except Exception as e:
            logger.error(f"Error verifying audit chain: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def get_audit_trail(
        self,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[AuditRecord]:
        """
        Retrieve audit trail with optional filtering
        """
        try:
            filtered_records = self.audit_chain.copy()
            
            if entity_id:
                filtered_records = [r for r in filtered_records if r.entity_id == entity_id]
            
            if entity_type:
                filtered_records = [r for r in filtered_records if r.entity_type == entity_type]
            
            if user_id:
                filtered_records = [r for r in filtered_records if r.user_id == user_id]
            
            if start_date:
                filtered_records = [r for r in filtered_records if r.timestamp >= start_date]
            
            if end_date:
                filtered_records = [r for r in filtered_records if r.timestamp <= end_date]
            
            # Sort by timestamp (newest first)
            filtered_records.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered_records
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {str(e)}")
            return []
    
    def export_audit_report(self, format: str = "json") -> str:
        """
        Export audit trail as a compliance report
        """
        try:
            if format.lower() == "json":
                report_data = {
                    "report_generated": datetime.utcnow().isoformat(),
                    "total_records": len(self.audit_chain),
                    "chain_verification": self.verify_audit_chain(),
                    "audit_records": [asdict(record) for record in self.audit_chain]
                }
                return json.dumps(report_data, indent=2)
            
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "Record ID", "Timestamp", "User ID", "Action", "Entity Type",
                    "Entity ID", "Document ID", "IP Address", "Data Hash", "Previous Hash"
                ])
                
                # Write records
                for record in self.audit_chain:
                    writer.writerow([
                        record.record_id, record.timestamp, record.user_id,
                        record.action, record.entity_type, record.entity_id,
                        record.document_id, record.ip_address, record.data_hash,
                        record.previous_hash
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting audit report: {str(e)}")
            raise

# Global instance
blockchain_audit_service = BlockchainAuditService() 