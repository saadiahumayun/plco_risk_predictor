# app/core/security.py
"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
security = HTTPBearer()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm="HS256"
    )
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """
    Get current user from JWT token.
    
    This is optional - only enforced if the endpoint uses this dependency.
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "is_admin": payload.get("is_admin", False)
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[Dict]:
    """Get current user if token is provided, None otherwise."""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def verify_api_key(api_key: str) -> bool:
    """Verify API key (simplified version)."""
    # In production, check against database
    # For now, just check format
    return len(api_key) == 32 and api_key.isalnum()


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests: int = 100, window: int = 3600):
        self.requests = requests
        self.window = window
        self.clients = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = datetime.utcnow()
        
        # Clean old entries
        self.clients = {
            cid: times for cid, times in self.clients.items()
            if any(t > now - timedelta(seconds=self.window) for t in times)
        }
        
        # Check rate limit
        if client_id not in self.clients:
            self.clients[client_id] = [now]
            return True
        
        # Filter recent requests
        recent = [
            t for t in self.clients[client_id]
            if t > now - timedelta(seconds=self.window)
        ]
        
        if len(recent) < self.requests:
            self.clients[client_id] = recent + [now]
            return True
        
        return False


# Create rate limiter instance
rate_limiter = RateLimiter(requests=100, window=3600)  # 100 requests per hour