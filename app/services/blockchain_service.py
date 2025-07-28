"""
Blockchain Audit Trail Service
Provides immutable legal document and compliance audit logging
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from web3 import Web3
from eth_account import Account
from solcx import compile_source
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.database.models.models import Document, User, AuditLog
from app.config import settings
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, Optional, Union
from solders.keypair import Keypair
from blockchain.audit_trail_integration import AuditTrailIntegration

logger = logging.getLogger(__name__)

class BlockchainRecord(BaseModel):
    transaction_hash: str
    block_number: int
    document_id: str
    user_id: str
    action: str
    timestamp: datetime
    data_hash: str

class BlockchainAuditService:
    def __init__(self):
        self.w3 = None
        self.contract = None
        self.account = None
        self._initialize_blockchain()
        
    def _initialize_blockchain(self):
        """Initialize blockchain connection and smart contract"""
        try:
            # Connect to blockchain network
            rpc_url = settings.BLOCKCHAIN_RPC_URL or "http://localhost:8545"
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            if not self.w3.is_connected():
                logger.warning("Blockchain connection failed, audit trail will use database only")
                return
                
            # Setup account
            private_key = settings.BLOCKCHAIN_PRIVATE_KEY
            if private_key:
                self.account = Account.from_key(private_key)
                logger.info(f"Blockchain account initialized: {self.account.address}")
            
            # Load or deploy contract
            self._setup_contract()
            
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {str(e)}")
            self.w3 = None
    
    def _setup_contract(self):
        """Setup the legal audit smart contract"""
        try:
            contract_address = settings.AUDIT_CONTRACT_ADDRESS
            
            if contract_address and self._is_valid_address(contract_address):
                # Load existing contract
                with open("contracts/LegalAudit_abi.json", "r") as f:
                    contract_abi = json.load(f)
                self.contract = self.w3.eth.contract(
                    address=contract_address, 
                    abi=contract_abi
                )
                logger.info(f"Loaded existing contract at {contract_address}")
            else:
                # Deploy new contract
                contract_address = self._deploy_contract()
                if contract_address:
                    logger.info(f"Deployed new contract at {contract_address}")
                    
        except Exception as e:
            logger.error(f"Contract setup failed: {str(e)}")
    
    def _deploy_contract(self) -> Optional[str]:
        """Deploy the legal audit smart contract"""
        try:
            # Smart contract source code
            contract_source_code = '''
            pragma solidity ^0.8.0;

            contract LegalAudit {
                struct AuditRecord {
                    string documentId;
                    string userId;
                    string action;
                    string dataHash;
                    uint256 timestamp;
                    address recorder;
                }
                
                mapping(uint256 => AuditRecord) public records;
                uint256 public recordCount;
                
                event AuditRecorded(
                    uint256 indexed recordId,
                    string documentId,
                    string userId,
                    string action,
                    uint256 timestamp
                );
                
                function recordAudit(
                    string memory _documentId,
                    string memory _userId,
                    string memory _action,
                    string memory _dataHash
                ) public returns (uint256) {
                    uint256 recordId = recordCount;
                    records[recordId] = AuditRecord({
                        documentId: _documentId,
                        userId: _userId,
                        action: _action,
                        dataHash: _dataHash,
                        timestamp: block.timestamp,
                        recorder: msg.sender
                    });
                    
                    recordCount++;
                    
                    emit AuditRecorded(recordId, _documentId, _userId, _action, block.timestamp);
                    return recordId;
                }
                
                function getRecord(uint256 _recordId) public view returns (
                    string memory documentId,
                    string memory userId,
                    string memory action,
                    string memory dataHash,
                    uint256 timestamp,
                    address recorder
                ) {
                    AuditRecord memory record = records[_recordId];
                    return (
                        record.documentId,
                        record.userId,
                        record.action,
                        record.dataHash,
                        record.timestamp,
                        record.recorder
                    );
                }
                
                function verifyRecord(
                    uint256 _recordId,
                    string memory _documentId,
                    string memory _dataHash
                ) public view returns (bool) {
                    AuditRecord memory record = records[_recordId];
                    return (
                        keccak256(bytes(record.documentId)) == keccak256(bytes(_documentId)) &&
                        keccak256(bytes(record.dataHash)) == keccak256(bytes(_dataHash))
                    );
                }
            }
            '''
            
            # Compile contract
            compiled_sol = compile_source(contract_source_code)
            contract_interface = compiled_sol['<stdin>:LegalAudit']
            
            # Deploy contract
            contract = self.w3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']
            )
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            transaction = contract.constructor().build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': 3000000,
                'gasPrice': self.w3.to_wei('20', 'gwei')
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Save contract info
            self.contract = self.w3.eth.contract(
                address=receipt.contractAddress,
                abi=contract_interface['abi']
            )
            
            # Save ABI and address for future use
            os.makedirs("contracts", exist_ok=True)
            with open("contracts/LegalAudit_abi.json", "w") as f:
                json.dump(contract_interface['abi'], f, indent=2)
            
            with open("contracts/LegalAudit_address.txt", "w") as f:
                f.write(receipt.contractAddress)
                
            return receipt.contractAddress
            
        except Exception as e:
            logger.error(f"Contract deployment failed: {str(e)}")
            return None
    
    async def record_audit(
        self,
        document_id: str,
        user_id: str,
        action: str,
        additional_data: Dict[str, Any] = None,
        session: AsyncSession = None
    ) -> Optional[BlockchainRecord]:
        """Record an audit event both on blockchain and database"""
        try:
            # Create data hash
            data_to_hash = {
                "document_id": document_id,
                "user_id": user_id,
                "action": action,
                "timestamp": datetime.utcnow().isoformat(),
                "additional_data": additional_data or {}
            }
            data_hash = hashlib.sha256(
                json.dumps(data_to_hash, sort_keys=True).encode()
            ).hexdigest()
            
            blockchain_record = None
            
            # Record on blockchain if available
            if self.w3 and self.contract and self.account:
                try:
                    blockchain_record = await self._record_on_blockchain(
                        document_id, user_id, action, data_hash
                    )
                except Exception as e:
                    logger.error(f"Blockchain recording failed: {str(e)}")
            
            # Always record in database as backup
            await self._record_in_database(
                document_id, user_id, action, data_hash, 
                additional_data, blockchain_record, session
            )
            
            return blockchain_record
            
        except Exception as e:
            logger.error(f"Audit recording failed: {str(e)}")
            return None
    
    async def _record_on_blockchain(
        self, document_id: str, user_id: str, action: str, data_hash: str
    ) -> BlockchainRecord:
        """Record audit on blockchain"""
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        # Build transaction
        transaction = self.contract.functions.recordAudit(
            document_id, user_id, action, data_hash
        ).build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': self.w3.to_wei('20', 'gwei')
        })
        
        # Sign and send
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return BlockchainRecord(
            transaction_hash=receipt.transactionHash.hex(),
            block_number=receipt.blockNumber,
            document_id=document_id,
            user_id=user_id,
            action=action,
            timestamp=datetime.utcnow(),
            data_hash=data_hash
        )
    
    async def _record_in_database(
        self,
        document_id: str,
        user_id: str,
        action: str,
        data_hash: str,
        additional_data: Dict[str, Any],
        blockchain_record: Optional[BlockchainRecord],
        session: AsyncSession
    ):
        """Record audit in database"""
        if not session:
            return
            
        audit_log = AuditLog(
            document_id=document_id,
            user_id=user_id,
            action=action,
            data_hash=data_hash,
            additional_data=additional_data,
            blockchain_tx_hash=blockchain_record.transaction_hash if blockchain_record else None,
            blockchain_block_number=blockchain_record.block_number if blockchain_record else None,
            timestamp=datetime.utcnow()
        )
        
        session.add(audit_log)
        await session.commit()
    
    async def get_audit_trail(
        self, 
        document_id: str = None, 
        user_id: str = None,
        session: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail for document or user"""
        if not session:
            return []
            
        query = select(AuditLog)
        
        if document_id:
            query = query.where(AuditLog.document_id == document_id)
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
            
        query = query.order_by(AuditLog.timestamp.desc())
        
        result = await session.execute(query)
        audit_logs = result.scalars().all()
        
        return [
            {
                "id": log.id,
                "document_id": log.document_id,
                "user_id": log.user_id,
                "action": log.action,
                "data_hash": log.data_hash,
                "additional_data": log.additional_data,
                "blockchain_tx_hash": log.blockchain_tx_hash,
                "blockchain_block_number": log.blockchain_block_number,
                "timestamp": log.timestamp,
                "verified": await self._verify_blockchain_record(log) if log.blockchain_tx_hash else False
            }
            for log in audit_logs
        ]
    
    async def _verify_blockchain_record(self, audit_log: AuditLog) -> bool:
        """Verify audit record against blockchain"""
        if not self.contract or not audit_log.blockchain_tx_hash:
            return False
            
        try:
            # Get transaction receipt
            receipt = self.w3.eth.get_transaction_receipt(audit_log.blockchain_tx_hash)
            
            # Verify the transaction was successful
            if receipt.status != 1:
                return False
            
            # Additional verification could be added here
            return True
            
        except Exception as e:
            logger.error(f"Blockchain verification failed: {str(e)}")
            return False
    
    def _is_valid_address(self, address: str) -> bool:
        """Check if address is valid Ethereum address"""
        try:
            return self.w3.is_address(address)
        except:
            return False

# Global blockchain service instance
blockchain_service = BlockchainAuditService() 

class BlockchainService:
    """Service for handling blockchain-based audit trail operations"""
    
    def __init__(self):
        """Initialize the blockchain service"""
        self.audit_trail = AuditTrailIntegration(config_path="blockchain/audit_trail_config.json")
        self.logger = logging.getLogger("BlockchainService")
    
    async def track_document_creation(self,
                                    file_path: Union[str, Path],
                                    user_keypair: Keypair,
                                    department: str,
                                    classification: str = "Confidential",
                                    **metadata) -> Optional[Dict[str, Any]]:
        """
        Track the creation of a new document in the blockchain
        
        Args:
            file_path: Path to the document
            user_keypair: User's Solana keypair
            department: Department the document belongs to
            classification: Document classification level
            **metadata: Additional metadata about the document
        
        Returns:
            Dict containing the audit entry or None if failed
        """
        try:
            # Calculate content hash
            from app.utils.blockchain_utils import calculate_content_hash
            content_hash = calculate_content_hash(file_path)
            if not content_hash:
                raise ValueError("Failed to calculate content hash")

            # Generate a deterministic transaction hash if needed
            tx_hash = hashlib.sha256(
                f"{content_hash}:{user_keypair.pubkey()}:{int(datetime.now().timestamp())}".encode()
            ).hexdigest()

            # Clean metadata to avoid duplicates
            metadata_copy = metadata.copy()
            for key in ['document_type', 'department', 'classification', 'content_hash', 'transaction_hash']:
                metadata_copy.pop(key, None)

            # Track in blockchain
            entry = self.audit_trail.track_new_document(
                file_path=file_path,
                user_keypair=user_keypair,
                department=department,
                classification=classification,
                content_hash=content_hash,
                transaction_hash=tx_hash,
                **metadata_copy  # Use cleaned metadata
            )
            
            if entry:
                # Ensure consistent field names
                entry["transaction_hash"] = (
                    entry.get("tx_hash") or 
                    entry.get("transaction_hash") or 
                    tx_hash  # Use generated hash as fallback
                )
                if "tx_hash" in entry:
                    del entry["tx_hash"]
                
                # Ensure content hash is included
                if "content_hash" not in entry:
                    entry["content_hash"] = content_hash

                # Log the entry for debugging
                self.logger.debug(f"Blockchain entry created: {entry}")
                    
            return entry
        except Exception as e:
            self.logger.error(f"Failed to track document creation: {e}")
            return None
    
    async def track_document_update(self,
                                  file_path: Union[str, Path],
                                  user_keypair: Keypair,
                                  modification_type: str,
                                  previous_version: str,
                                  change_description: str,
                                  **metadata) -> Optional[Dict[str, Any]]:
        """
        Track document updates in the blockchain
        
        Args:
            file_path: Path to the document
            user_keypair: User's Solana keypair
            modification_type: Type of modification made
            previous_version: Previous version number
            change_description: Description of changes made
            **metadata: Additional metadata about the update
        """
        try:
            # Calculate content hash
            from app.utils.blockchain_utils import calculate_content_hash
            content_hash = calculate_content_hash(file_path)
            if not content_hash:
                raise ValueError("Failed to calculate content hash")

            # Generate a deterministic transaction hash if needed
            tx_hash = hashlib.sha256(
                f"{content_hash}:{user_keypair.pubkey()}:{modification_type}:{int(datetime.now().timestamp())}".encode()
            ).hexdigest()

            entry = self.audit_trail.track_document_update(
                file_path=file_path,
                user_keypair=user_keypair,
                modification_type=modification_type,
                previous_version=previous_version,
                change_description=change_description,
                content_hash=content_hash,
                transaction_hash=tx_hash,  # Provide default transaction hash
                **metadata
            )
            
            if entry:
                # Ensure consistent field names
                entry["transaction_hash"] = (
                    entry.get("tx_hash") or 
                    entry.get("transaction_hash") or 
                    tx_hash  # Use generated hash as fallback
                )
                if "tx_hash" in entry:
                    del entry["tx_hash"]
                
                # Ensure content hash is included
                if "content_hash" not in entry:
                    entry["content_hash"] = content_hash

                # Log the entry for debugging
                self.logger.debug(f"Blockchain entry updated: {entry}")
                    
            return entry
        except Exception as e:
            self.logger.error(f"Failed to track document update: {e}")
            return None
    
    async def track_document_access(self,
                                  file_path: Union[str, Path],
                                  user_keypair: Keypair,
                                  access_type: str,
                                  purpose: str,
                                  **metadata) -> Optional[Dict[str, Any]]:
        """
        Track document access in the blockchain
        
        Args:
            file_path: Path to the document
            user_keypair: User's Solana keypair
            access_type: Type of access (READ, WRITE, etc.)
            purpose: Purpose of accessing the document
            **metadata: Additional metadata about the access
        """
        try:
            # Calculate content hash
            from app.utils.blockchain_utils import calculate_content_hash
            content_hash = calculate_content_hash(file_path)
            if not content_hash:
                raise ValueError("Failed to calculate content hash")

            # Generate a deterministic transaction hash if needed
            tx_hash = hashlib.sha256(
                f"{content_hash}:{user_keypair.pubkey()}:{access_type}:{int(datetime.now().timestamp())}".encode()
            ).hexdigest()

            entry = self.audit_trail.track_document_access(
                file_path=file_path,
                user_keypair=user_keypair,
                access_type=access_type,
                purpose=purpose,
                content_hash=content_hash,
                transaction_hash=tx_hash,  # Provide default transaction hash
                **metadata
            )
            
            if entry:
                # Ensure consistent field names
                entry["transaction_hash"] = (
                    entry.get("tx_hash") or 
                    entry.get("transaction_hash") or 
                    tx_hash  # Use generated hash as fallback
                )
                if "tx_hash" in entry:
                    del entry["tx_hash"]
                
                # Ensure content hash is included
                if "content_hash" not in entry:
                    entry["content_hash"] = content_hash

                # Log the entry for debugging
                self.logger.debug(f"Blockchain entry accessed: {entry}")
                    
            return entry
        except Exception as e:
            self.logger.error(f"Failed to track document access: {e}")
            return None
    
    async def get_document_history(self, file_path: Union[str, Path]) -> Optional[list]:
        """
        Get the complete audit history for a document from the blockchain
        
        Args:
            file_path: Path to the document
        
        Returns:
            List of audit trail entries or None if failed
        """
        try:
            history = self.audit_trail.get_document_audit_history(file_path)
            return history
        except Exception as e:
            self.logger.error(f"Failed to get document history: {e}")
            return None
    
    async def verify_document_integrity(self, file_path: Union[str, Path], expected_hash: str) -> bool:
        """
        Verify the integrity of a document using its blockchain hash
        
        Args:
            file_path: Path to the document
            expected_hash: Expected document hash from the blockchain
        
        Returns:
            bool: True if document is valid, False otherwise
        """
        try:
            is_valid = self.audit_trail.verify_document(file_path, expected_hash)
            return is_valid
        except Exception as e:
            self.logger.error(f"Failed to verify document: {e}")
            return False 