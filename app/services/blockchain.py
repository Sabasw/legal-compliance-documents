import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from web3 import Web3
# Note: geth_poa_middleware import removed due to version compatibility

from simple_config import settings

logger = logging.getLogger(__name__)

class BlockchainLogger:
    def __init__(self):
        try:
            self.w3 = Web3(Web3.HTTPProvider(settings.BLOCKCHAIN_PROVIDER_URL))
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to blockchain network")

            # Note: PoA middleware disabled for compatibility

            with open('contract_abi.json') as f:
                self.contract_abi = json.load(f)

            self.contract = self.w3.eth.contract(
                address=settings.BLOCKCHAIN_CONTRACT_ADDRESS,
                abi=self.contract_abi
            )
            self.account = self.w3.eth.account.from_key(os.getenv('BLOCKCHAIN_PRIVATE_KEY'))
            logger.info("Blockchain logger initialized")
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {str(e)}")
            raise RuntimeError("Blockchain integration unavailable")

    def log_compliance_result(self, doc_hash: str, verdict: str, jurisdiction: str) -> Dict[str, Any]:
        """Log compliance result to blockchain with retry logic and confirmation"""
        for attempt in range(settings.BLOCKCHAIN_MAX_RETRIES):
            try:
                nonce = self.w3.eth.get_transaction_count(self.account.address)
                tx = self.contract.functions.recordCompliance(
                    doc_hash,
                    verdict,
                    jurisdiction,
                    int(datetime.now().timestamp())
                ).build_transaction({
                    'chainId': 1,  # Mainnet
                    'gas': settings.BLOCKCHAIN_GAS_LIMIT,
                    'gasPrice': self.w3.to_wei(settings.BLOCKCHAIN_GAS_PRICE, 'gwei'),
                    'nonce': nonce,
                })

                signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

                # Wait for transaction receipt with confirmations
                receipt = None
                for _ in range(30):  # 30 second timeout
                    receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                    if receipt is not None and receipt.blockNumber is not None:
                        current_block = self.w3.eth.block_number
                        if current_block - receipt.blockNumber >= settings.BLOCKCHAIN_CONFIRMATION_BLOCKS:
                            break
                    time.sleep(1)

                if receipt and receipt.status == 1:
                    logger.info(f"Compliance result logged to blockchain: {tx_hash.hex()}")
                    return {
                        "tx_hash": tx_hash.hex(),
                        "block_number": receipt.blockNumber,
                        "gas_used": receipt.gasUsed,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    raise ValueError("Transaction failed or not confirmed")

            except Exception as e:
                logger.warning(f"Blockchain log attempt {attempt + 1} failed: {str(e)}")
                if attempt == settings.BLOCKCHAIN_MAX_RETRIES - 1:
                    logger.error("Max retries reached for blockchain logging")
                    raise

    def verify_compliance_record(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Verify a compliance record exists on-chain"""
        try:
            record = self.contract.functions.getComplianceRecord(doc_hash).call()
            if record and len(record) >= 4:  # Ensure we have all expected fields
                return {
                    "verdict": record[1],
                    "jurisdiction": record[2],
                    "timestamp": datetime.fromtimestamp(record[3]).isoformat(),
                    "block_number": record[4] if len(record) > 4 else None
                }
            return None
        except Exception as e:
            logger.error(f"Blockchain verification failed: {str(e)}")
            return None 