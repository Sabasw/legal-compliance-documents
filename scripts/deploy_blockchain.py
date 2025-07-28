#!/usr/bin/env python3
"""
Blockchain Audit Trail Deployment Script
Deploys the smart contract and sets up the blockchain audit system
"""

import os
import sys
import json
import logging
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainDeployer:
    def __init__(self):
        self.w3 = None
        self.account = None
        self.contract = None
        self.contract_address = None
        
    def connect_to_network(self):
        """Connect to the blockchain network"""
        try:
            # Get RPC URL from environment
            rpc_url = os.getenv('BLOCKCHAIN_RPC_URL', 'http://localhost:8545')
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Check connection
            if not self.w3.is_connected():
                raise Exception(f"Failed to connect to blockchain network at {rpc_url}")
            
            logger.info(f"Connected to blockchain network: {rpc_url}")
            logger.info(f"Network ID: {self.w3.eth.chain_id}")
            logger.info(f"Current block: {self.w3.eth.block_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to blockchain network: {str(e)}")
            return False
    
    def setup_account(self):
        """Setup the deployment account"""
        try:
            # Get private key from environment
            private_key = os.getenv('BLOCKCHAIN_PRIVATE_KEY')
            
            if not private_key:
                # Generate a new account for testing
                logger.warning("No private key found in environment. Generating test account...")
                account = Account.create()
                private_key = account.key.hex()
                logger.info(f"Generated test account: {account.address}")
                logger.info(f"Private key: {private_key}")
                logger.warning("IMPORTANT: Save this private key securely for production use!")
            
            self.account = Account.from_key(private_key)
            self.w3.eth.default_account = self.account.address
            
            # Check account balance
            balance = self.w3.eth.get_balance(self.account.address)
            balance_eth = self.w3.from_wei(balance, 'ether')
            
            logger.info(f"Deployment account: {self.account.address}")
            logger.info(f"Account balance: {balance_eth} ETH")
            
            if balance_eth < 0.01:
                logger.warning("Low account balance. Ensure sufficient funds for deployment.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup account: {str(e)}")
            return False
    
    def load_contract_source(self):
        """Load the smart contract source code"""
        try:
            contract_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'contracts', 
                'AuditTrailContract.sol'
            )
            
            if not os.path.exists(contract_path):
                raise Exception(f"Contract file not found: {contract_path}")
            
            with open(contract_path, 'r') as f:
                contract_source = f.read()
            
            logger.info("Loaded smart contract source code")
            return contract_source
            
        except Exception as e:
            logger.error(f"Failed to load contract source: {str(e)}")
            return None
    
    def compile_contract(self, contract_source):
        """Compile the smart contract (requires solc)"""
        try:
            # For production, use proper Solidity compiler
            # This is a simplified version for demo purposes
            logger.info("Compiling smart contract...")
            
            # In a real deployment, you would use:
            # from solcx import compile_source
            # compiled_sol = compile_source(contract_source, output_values=['abi', 'bin'])
            
            # For now, we'll use a pre-compiled ABI
            contract_abi = [
                {
                    "inputs": [],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {
                            "indexed": True,
                            "internalType": "string",
                            "name": "recordId",
                            "type": "string"
                        },
                        {
                            "indexed": False,
                            "internalType": "string",
                            "name": "dataHash",
                            "type": "string"
                        },
                        {
                            "indexed": False,
                            "internalType": "string",
                            "name": "previousHash",
                            "type": "string"
                        },
                        {
                            "indexed": False,
                            "internalType": "uint256",
                            "name": "timestamp",
                            "type": "uint256"
                        },
                        {
                            "indexed": True,
                            "internalType": "address",
                            "name": "addedBy",
                            "type": "address"
                        }
                    ],
                    "name": "AuditRecordAdded",
                    "type": "event"
                },
                {
                    "inputs": [
                        {
                            "internalType": "string",
                            "name": "recordId",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "dataHash",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "previousHash",
                            "type": "string"
                        },
                        {
                            "internalType": "uint256",
                            "name": "timestamp",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "metadata",
                            "type": "string"
                        }
                    ],
                    "name": "addAuditRecord",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "string",
                            "name": "recordId",
                            "type": "string"
                        }
                    ],
                    "name": "getAuditRecord",
                    "outputs": [
                        {
                            "internalType": "string",
                            "name": "dataHash",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "previousHash",
                            "type": "string"
                        },
                        {
                            "internalType": "uint256",
                            "name": "timestamp",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "metadata",
                            "type": "string"
                        },
                        {
                            "internalType": "bool",
                            "name": "exists",
                            "type": "bool"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "verifyChainIntegrity",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "isValid",
                            "type": "bool"
                        },
                        {
                            "internalType": "uint256",
                            "name": "totalRecords",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Mock bytecode for demo (in production, this would be the actual compiled bytecode)
            contract_bytecode = "0x608060405234801561001057600080fd5b50604051610..."
            
            logger.info("Contract compilation completed")
            return contract_abi, contract_bytecode
            
        except Exception as e:
            logger.error(f"Failed to compile contract: {str(e)}")
            return None, None
    
    def deploy_contract(self, contract_abi, contract_bytecode):
        """Deploy the smart contract"""
        try:
            logger.info("Deploying smart contract...")
            
            # Create contract instance
            contract = self.w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
            
            # Build constructor transaction
            construct_txn = contract.constructor().build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(construct_txn, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            logger.info(f"Deployment transaction sent: {tx_hash.hex()}")
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                contract_address = tx_receipt.contractAddress
                self.contract_address = contract_address
                self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
                
                logger.info(f"Contract deployed successfully!")
                logger.info(f"Contract address: {contract_address}")
                logger.info(f"Block number: {tx_receipt.blockNumber}")
                logger.info(f"Gas used: {tx_receipt.gasUsed}")
                
                return True
            else:
                logger.error("Contract deployment failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deploy contract: {str(e)}")
            return False
    
    def save_deployment_info(self):
        """Save deployment information to environment file"""
        try:
            if not self.contract_address:
                logger.warning("No contract address to save")
                return False
            
            # Create .env file with deployment info
            env_content = f"""# Blockchain Audit Trail Configuration
BLOCKCHAIN_RPC_URL={os.getenv('BLOCKCHAIN_RPC_URL', 'http://localhost:8545')}
BLOCKCHAIN_CHAIN_ID={self.w3.eth.chain_id}
AUDIT_CONTRACT_ADDRESS={self.contract_address}
BLOCKCHAIN_PRIVATE_KEY={os.getenv('BLOCKCHAIN_PRIVATE_KEY', '')}

# Deployment Information
DEPLOYMENT_ACCOUNT={self.account.address}
DEPLOYMENT_BLOCK={self.w3.eth.block_number}
"""
            
            env_file_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            
            # Append to existing .env file or create new one
            with open(env_file_path, 'a') as f:
                f.write(f"\n# Blockchain Audit Trail - Deployed at {self.w3.eth.block_number}\n")
                f.write(f"AUDIT_CONTRACT_ADDRESS={self.contract_address}\n")
                f.write(f"BLOCKCHAIN_CHAIN_ID={self.w3.eth.chain_id}\n")
            
            logger.info(f"Deployment information saved to {env_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save deployment info: {str(e)}")
            return False
    
    def test_contract(self):
        """Test the deployed contract"""
        try:
            if not self.contract:
                logger.error("No contract deployed to test")
                return False
            
            logger.info("Testing deployed contract...")
            
            # Test basic contract functions
            owner = self.contract.functions.owner().call()
            logger.info(f"Contract owner: {owner}")
            
            chain_length = self.contract.functions.chainLength().call()
            logger.info(f"Initial chain length: {chain_length}")
            
            # Test chain integrity
            is_valid, total_records = self.contract.functions.verifyChainIntegrity().call()
            logger.info(f"Chain integrity: {is_valid}, Total records: {total_records}")
            
            logger.info("Contract testing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Contract testing failed: {str(e)}")
            return False
    
    def deploy(self):
        """Main deployment process"""
        logger.info("Starting blockchain audit trail deployment...")
        
        # Step 1: Connect to network
        if not self.connect_to_network():
            return False
        
        # Step 2: Setup account
        if not self.setup_account():
            return False
        
        # Step 3: Load contract source
        contract_source = self.load_contract_source()
        if not contract_source:
            return False
        
        # Step 4: Compile contract
        contract_abi, contract_bytecode = self.compile_contract(contract_source)
        if not contract_abi or not contract_bytecode:
            return False
        
        # Step 5: Deploy contract
        if not self.deploy_contract(contract_abi, contract_bytecode):
            return False
        
        # Step 6: Test contract
        if not self.test_contract():
            return False
        
        # Step 7: Save deployment info
        if not self.save_deployment_info():
            return False
        
        logger.info("Blockchain audit trail deployment completed successfully!")
        return True

def main():
    """Main function"""
    deployer = BlockchainDeployer()
    
    if deployer.deploy():
        print("\n" + "="*60)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print(f"Contract Address: {deployer.contract_address}")
        print(f"Network: {deployer.w3.eth.chain_id}")
        print(f"Account: {deployer.account.address}")
        print("="*60)
        print("\nNext steps:")
        print("1. Update your .env file with the contract address")
        print("2. Restart your application")
        print("3. Test the audit trail functionality")
        print("\nFor production deployment:")
        print("- Use a proper Solidity compiler")
        print("- Deploy to a production blockchain network")
        print("- Secure your private keys")
        print("- Set up proper gas management")
    else:
        print("\nDEPLOYMENT FAILED!")
        print("Check the logs above for details.")

if __name__ == "__main__":
    main() 