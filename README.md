# 🚀 SolveLex Enhanced File Tracking System

A comprehensive legal document management system with blockchain-based audit trails, AI-powered analysis, and advanced file integrity tracking.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [File Tracking Guide](#-file-tracking-guide)
- [Blockchain Integration](#-blockchain-integration)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔍 **Advanced File Tracking**
- **File Integrity Verification**: SHA-256 hash-based integrity checking
- **Duplicate Detection**: Identify and prevent duplicate file uploads
- **Version History**: Complete audit trail of all document versions
- **Change Detection**: Detect file modifications and track changes
- **Blockchain Verification**: Immutable audit trail on blockchain

### 🔗 **Blockchain Integration**
- **Ethereum Support**: Web3 integration for Ethereum blockchain
- **Solana Support**: Solana blockchain integration
- **Smart Contract Audit**: Legal document audit trails
- **Transaction Verification**: Blockchain transaction verification
- **Immutable Records**: Tamper-proof document records

### 🤖 **AI-Powered Analysis**
- **Document Classification**: Automatic document type detection
- **Compliance Analysis**: Legal compliance checking
- **Risk Assessment**: Automated risk scoring
- **Content Extraction**: Advanced text extraction from documents
- **Intelligent Summarization**: AI-generated document summaries

### 📊 **Comprehensive Reporting**
- **PDF Report Generation**: Structured PDF reports with analysis
- **Visualization**: Risk profiles and decision factor charts
- **Audit Trails**: Complete activity logging
- **Export Capabilities**: Multiple export formats
- **Custom Reports**: Configurable report templates

### 🔐 **Security & Compliance**
- **JWT Authentication**: Secure user authentication
- **Role-Based Access**: User role management
- **Data Encryption**: End-to-end encryption
- **Audit Logging**: Complete activity audit trails
- **GDPR Compliance**: Privacy and data protection

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- 4GB RAM minimum (8GB recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd solvelex-updated

# Create virtual environment
python -m venv solvelex_env
source solvelex_env/bin/activate  # On Windows: solvelex_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

### Test the Installation

```bash
# Health check
curl http://localhost:8080/health

# API documentation
open http://localhost:8080/docs
```

## 📦 Installation

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.8 or higher |
| **PostgreSQL** | 12 or higher |
| **RAM** | 4GB minimum (8GB recommended) |
| **Storage** | 10GB free space |
| **OS** | Windows 10/11, macOS, Linux |

### Dependencies

#### Core Framework
```bash
pip install fastapi==0.110.3 uvicorn==0.15.0 pydantic==1.10.22
```

#### Database
```bash
pip install SQLAlchemy==1.4.23 psycopg2==2.9.10
```

#### Blockchain
```bash
pip install web3==7.12.0 eth-account==0.13.7 solana==0.36.7
```

#### AI & ML
```bash
pip install groq==0.29.0 sentence-transformers==5.0.0 torch==2.7.1
```

#### File Processing
```bash
pip install pdfplumber==0.11.7 PyPDF2==3.0.1 reportlab==4.1.0
```

### Database Setup

```sql
-- Create database and user
CREATE DATABASE solvelex_db;
CREATE USER solvelex_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE solvelex_db TO solvelex_user;
```

### Environment Configuration

Create a `.env` file:

```env
# Database Configuration
DATABASE_URL=postgresql://solvelex_user:your_password@localhost:5432/solvelex_db

# Blockchain Configuration
BLOCKCHAIN_RPC_URL=http://localhost:8545
BLOCKCHAIN_PRIVATE_KEY=your_private_key
AUDIT_CONTRACT_ADDRESS=your_contract_address

# AI Configuration
GROQ_API_KEY=your_groq_api_key

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=uploads
REPORTS_DIR=reports

# Server Configuration
HOST=0.0.0.0
PORT=8080
DEBUG=True
```

## 📚 API Documentation

### Core Endpoints

#### Document Upload
```bash
POST /documents/upload/
```
Upload and analyze legal documents with blockchain tracking.

**Response includes:**
- File integrity information
- Blockchain transaction details
- AI analysis results
- Risk assessment
- Compliance status

#### Document Retrieval
```bash
GET /documents/{doc_id}
```
Retrieve document with complete tracking information.

#### File Integrity Verification
```bash
GET /documents/{doc_id}/integrity
```
Comprehensive file integrity verification and tracking details.

#### Document Tracking
```bash
GET /documents/{doc_id}/tracking
```
Complete document tracking and version history.

### Enhanced Response Format

```json
{
  "doc_id": "ab638807-8789-4c32-ac69-2a48de9887c5",
  "filename": "legal_document.pdf",
  
  "file_integrity": {
    "file_size_bytes": 479749,
    "file_size_mb": 0.46,
    "content_hash": "abbe7f108461a35fadd4d9d92b2eb88aacd2891de3e5838b916cd5206aa1c7ed",
    "hash_algorithm": "SHA-256",
    "integrity_status": "VERIFIED",
    "is_duplicate": false,
    "duplicate_count": 0
  },
  
  "blockchain": {
    "transaction_hash": "012793dc7a07ae462f745a88fcdc2640d648d50eccab0704a65ce5bd05cd56a4",
    "content_hash": "abbe7f108461a35fadd4d9d92b2eb88aacd2891de3e5838b916cd5206aa1c7ed",
    "blockchain_status": "CONFIRMED",
    "verification_url": "https://blockchain.example.com/tx/012793dc7a07ae462f745a88fcdc2640d648d50eccab0704a65ce5bd05cd56a4"
  },
  
  "change_detection": {
    "is_new_file": true,
    "is_modified_version": false,
    "content_changed": true,
    "hash_verification": "PASSED",
    "blockchain_verification": "PASSED"
  }
}
```

## 🔍 File Tracking Guide

### Understanding File Integrity

The system provides comprehensive file tracking with:

#### **File Integrity Status**
| Status | Meaning | Action Required |
|--------|---------|-----------------|
| `VERIFIED` | File content matches original hash | ✅ No action needed |
| `MODIFIED` | File content has changed since upload | ⚠️ Review file changes |
| `MISSING` | File no longer exists on disk | ❌ File needs to be restored |

#### **Change Detection**
- **New Files**: First-time uploads
- **Modified Versions**: Updated content detection
- **Duplicates**: Same content uploaded before
- **Hash Verification**: SHA-256 integrity checking

### Testing File Tracking

```bash
# Upload original file
curl -X POST 'http://localhost:8080/documents/upload/' \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -F 'file=@document.pdf' \
  -F 'department=legal' \
  -F 'classification=Confidential'

# Upload modified version
curl -X POST 'http://localhost:8080/documents/upload/' \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -F 'file=@modified_document.pdf' \
  -F 'department=legal' \
  -F 'classification=Confidential'

# Verify integrity
curl -X GET 'http://localhost:8080/documents/{doc_id}/integrity' \
  -H 'Authorization: Bearer YOUR_TOKEN'
```

## 🔗 Blockchain Integration

### Supported Blockchains

#### **Ethereum Integration**
- Web3.py integration
- Smart contract audit trails
- Transaction verification
- Gas optimization

#### **Solana Integration**
- Solana blockchain support
- Fast transaction processing
- Low-cost operations
- Program integration

### Blockchain Features

#### **Document Audit Trail**
```json
{
  "blockchain": {
    "transaction_hash": "0x1234...",
    "content_hash": "abbe7f10...",
    "block_number": 12345678,
    "timestamp": "2025-07-28T13:49:31.352731+05:00",
    "status": "CONFIRMED"
  }
}
```

#### **Smart Contract Integration**
- Legal document audit contracts
- Immutable record keeping
- Automated compliance checking
- Tamper-proof verification

### Testing Blockchain Features

```bash
# Test blockchain tracking
python test_blockchain_tracking.py

# Create test modifications
python modify_7_pdf.py

# Verify blockchain entries
curl -X GET 'http://localhost:8080/documents/{doc_id}/tracking' \
  -H 'Authorization: Bearer YOUR_TOKEN'
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   PostgreSQL    │
│   (React/Vue)   │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Blockchain    │
                       │   (Ethereum/    │
                       │   Solana)       │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │   (Groq/ML)     │
                       └─────────────────┘
```

### Key Modules

#### **Core Modules**
- `app/main.py`: FastAPI application entry point
- `app/api/`: REST API endpoints
- `app/database/`: Database models and connections
- `app/services/`: Business logic services

#### **File Tracking**
- `app/utils/file_organizer.py`: File organization and management
- `app/utils/blockchain_utils.py`: Blockchain utilities
- `app/utils/pdf_generator.py`: PDF report generation

#### **AI & Analysis**
- `app/core/compliance.py`: Compliance analysis engine
- `app/core/document.py`: Document processing
- `app/services/ai_service.py`: AI integration

#### **Blockchain**
- `app/services/blockchain_service.py`: Blockchain integration
- `app/database/models/blockchain_models.py`: Blockchain data models

### Data Flow

1. **Document Upload**: File uploaded via API
2. **Content Analysis**: AI analyzes document content
3. **Hash Calculation**: SHA-256 hash generated
4. **Blockchain Recording**: Transaction recorded on blockchain
5. **Database Storage**: Metadata stored in PostgreSQL
6. **Report Generation**: PDF reports with analysis
7. **Integrity Verification**: Ongoing hash verification

## 🧪 Testing

### Test Scripts

#### **Blockchain Tracking Test**
```bash
python test_blockchain_tracking.py
```

#### **File Modification Test**
```bash
python modify_7_pdf.py
```

#### **API Testing**
```bash
# Test file upload
curl -X POST 'http://localhost:8080/documents/upload/' \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -F 'file=@test.pdf' \
  -F 'department=legal' \
  -F 'classification=Confidential'

# Test integrity verification
curl -X GET 'http://localhost:8080/documents/{doc_id}/integrity' \
  -H 'Authorization: Bearer YOUR_TOKEN'
```

### Test Files

The system includes test files for verification:
- `knowledge_docs/7.pdf`: Original test document
- `uploads/7_modified_v*.pdf`: Modified test versions
- `test_blockchain_tracking.py`: Blockchain testing script
- `modify_7_pdf.py`: File modification script

## 📊 Performance

### Benchmarks

| Operation | Average Time | Notes |
|-----------|-------------|-------|
| **File Upload** | 2-5 seconds | Includes analysis and blockchain recording |
| **Hash Calculation** | <100ms | SHA-256 file hashing |
| **Blockchain Transaction** | 1-3 seconds | Network dependent |
| **AI Analysis** | 3-8 seconds | Document complexity dependent |
| **PDF Generation** | 1-2 seconds | Report complexity dependent |

### Optimization

#### **Database Indexes**
```sql
CREATE INDEX idx_documents_owner_id ON documents(owner_id);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_content_hash ON documents(blockchain_content_hash);
```

#### **Caching Strategy**
- Redis caching for frequently accessed data
- File hash caching for integrity checks
- Blockchain transaction caching

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `BLOCKCHAIN_RPC_URL` | Blockchain RPC endpoint | `http://localhost:8545` |
| `GROQ_API_KEY` | Groq AI API key | Required |
| `SECRET_KEY` | JWT secret key | Required |
| `UPLOAD_DIR` | File upload directory | `uploads` |
| `REPORTS_DIR` | Report storage directory | `reports` |

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn==21.2.0

# Start production server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080

# With SSL (recommended for production)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --certfile=cert.pem --keyfile=key.pem
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Write comprehensive tests

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_file_tracking.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

### Documentation
- [API Documentation](http://localhost:8080/docs)
- [File Tracking Guide](FILE_TRACKING_GUIDE.md)
- [Blockchain Testing Guide](BLOCKCHAIN_TRACKING_TEST.md)

### Issues
- [GitHub Issues](https://github.com/your-repo/issues)
- [Feature Requests](https://github.com/your-repo/issues/new)

### Community
- [Discord Server](https://discord.gg/solvelex)
- [Documentation](https://docs.solvelex.ai)

## 🎯 Roadmap

### Upcoming Features
- [ ] **Multi-language Support**: Internationalization
- [ ] **Advanced Analytics**: Business intelligence dashboard
- [ ] **Mobile App**: iOS and Android applications
- [ ] **API Rate Limiting**: Enhanced security
- [ ] **Real-time Notifications**: WebSocket integration
- [ ] **Advanced AI Models**: GPT-4 integration
- [ ] **Cloud Storage**: AWS S3 integration
- [ ] **Multi-tenant**: SaaS capabilities

### Performance Improvements
- [ ] **Caching Layer**: Redis integration
- [ ] **CDN Integration**: Global content delivery
- [ ] **Database Optimization**: Query optimization
- [ ] **Async Processing**: Background task processing

---

**Built with ❤️ by the SolveLex Team**

*Empowering legal professionals with AI-driven document management and blockchain security.* 