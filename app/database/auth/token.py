from fastapi import HTTPException
from jose import JWTError, jwt  # Use jose's jwt
from app.database.scehmas.schemas import TokenData
from datetime import datetime, timezone, timedelta
import secrets

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7" # We'll update this in future
# SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60

def create_access_token(data: dict):
    to_encode = data.copy()
    # print("Token data: ", to_encode)
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    # print("expire in minutes: ", expire)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    try:
        # Decode and verify the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check for expiration
        # exp = payload.get("exp")
        # if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(tz=timezone.utc):
        #     raise HTTPException(status_code=401, detail="Token has expired")

        # Extract user information (e.g., email)
        email = payload.get("sub")
        if email is None:
            raise credentials_exception

        return TokenData(email=email)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise credentials_exception