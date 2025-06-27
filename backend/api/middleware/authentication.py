# backend/api/middleware/authentication.py

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Dict, Any

from backend.utils.logger import Logger
from backend.api.routes.consciousness import get_consciousness 
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness

logger = Logger(__name__)

# This dependency scheme extracts the token from the "Authorization: Bearer <token>" header.
# The tokenUrl is not used for validation here but is required by FastAPI for OpenAPI spec generation.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") 

async def authenticate_request(
    token: str = Depends(oauth2_scheme),
    consciousness: UnifiedConsciousness = Depends(get_consciousness)
) -> Dict[str, Any]:
    """
    FastAPI dependency that authenticates a request by validating its JWT.
    
    This function is intended for use in the `dependencies` list of protected API routes.
    It automatically handles extracting the token from the Authorization header and raises
    an appropriate HTTP exception if authentication fails.
    
    Args:
        token (str): The JWT token extracted by FastAPI's security utility.
        consciousness (UnifiedConsciousness): The injected consciousness instance, used to access configuration.

    Raises:
        HTTPException(401): If the token is missing, malformed, invalid, or expired.

    Returns:
        Dict[str, Any]: The decoded payload of the token if it is valid. This can be used
                        by the endpoint for user-specific logic.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        api_config = consciousness.config.get('api', {})
        jwt_secret = api_config.get('jwt_secret')
        jwt_algorithm = api_config.get('jwt_algorithm')
        
        if not jwt_secret or not jwt_algorithm:
            logger.critical("JWT secret or algorithm is not configured in prometheus_config.yaml! Authentication is disabled.")
            raise credentials_exception

        # Decode the JWT. The `jwt.decode` function handles signature validation
        # and token expiration checks automatically.
        payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])
        
        # --- Optional: Add more claims validation here ---
        # For example, you could check for a specific 'sub' (subject) or 'scope'.
        # username: str = payload.get("sub")
        # if username is None:
        #     logger.warning("Authentication failed: 'sub' (subject) claim missing in JWT.")
        #     raise credentials_exception
        
        if not payload:
            logger.warning("Authentication failed: JWT payload is empty after decoding.")
            raise credentials_exception
            
    except JWTError as e:
        logger.warning(f"Authentication failed due to JWT Error: {e}")
        # Re-raise as a standard 401 exception to be sent to the client.
        raise credentials_exception
    except Exception as e:
        logger.error(f"An unexpected error occurred during token decoding: {e}", exc_info=True)
        raise credentials_exception
        
    logger.debug(f"Authentication successful for payload: {payload}")
    return payload