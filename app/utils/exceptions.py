# exceptions.py
from fastapi import HTTPException, status

class DocumentProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document processing error: {detail}"
        )

class AnalysisError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis error: {detail}"
        )

class KnowledgeBaseError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Knowledge base error: {detail}"
        )