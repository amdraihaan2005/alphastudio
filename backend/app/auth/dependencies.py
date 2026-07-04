from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase_auth.errors import AuthApiError
from app.database.supabase import supabase
from app.database.connection import get_db
from app.database.models.user import User
from sqlalchemy.orm import Session
import uuid

# Security scheme to retrieve the Authorization Bearer token header
security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency to extract and validate the Supabase JWT token from the Authorization header.
    Queries Supabase's Auth API to verify token signature and validity.
    Looks up the corresponding user in the local postgres database, syncing/creating the record if it doesn't exist yet.
    Raises HTTP 401 for invalid or expired credentials.
    """
    token = credentials.credentials
    try:
        # Validate JWT token using Supabase's auth service
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate authentication credentials"
            )
            
        supabase_user = user_response.user
        user_id = uuid.UUID(supabase_user.id)
        
        # Check if the user exists in our local SQL database
        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user:
            # Sync user information to local database on first authentication
            db_user = User(
                id=user_id,
                email=supabase_user.email or ""
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
        return db_user

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authentication token: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
