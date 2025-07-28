import jwt
import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException, status
import jwt.exceptions

# Dynamically generate a secure random SECRET_KEY (32 bytes of randomness)
SECRET_KEY = secrets.token_hex(32)  # This generates a 64-character hexadecimal string (256-bit key)
ALGORITHM = "HS256"

def create_jwt(data: dict, expires_delta: timedelta = None):
    """
    Creates a JWT token with an expiration time.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))  # Default to 30 minutes
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_jwt(token: str):
    """
    Decodes the JWT token and validates it.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
