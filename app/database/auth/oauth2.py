from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from app.database.auth.token import verify_token
from app.database.models.models import User  # Import your User model
from app.database.db.db_connection import get_db
# from app.endpoints.auth import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    """
    Extracts and verifies the token from the Authorization header or query parameter.
    Returns user data if the token is valid.
    """

    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    
    # Raise error if no token is provided in either the header or query parameter
    if not auth_header and not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing or invalid",
        )
    
    # Extract token from the Authorization header if present
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]

    # Create exception for invalid or expired tokens
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Verify the token
        token_data = verify_token(token, credentials_exception)
        # print("Token data: ", token_data)
        
        # Return the decoded token data (e.g., user email or ID)
        return {"isAuthenticated": True, "email": token_data.email}
    except HTTPException as e:
        # Handle explicit HTTP exceptions (e.g., invalid credentials)
        raise e
    except JWTError:
        # Handle generic JWT decoding errors
        raise credentials_exception
