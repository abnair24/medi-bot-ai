"""JWT encode/decode and FastAPI authentication dependency.

Uses PyJWT (import as `jwt`) for stateless token auth.
The `get_current_user` dependency is injected into protected endpoints via Depends().
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config.settings import settings
from src.models.auth import TokenPayload
from src.utils.exceptions import AuthenticationError

# Algorithm constant — HS256 is symmetric HMAC. Fine for a single-service demo.
# Production multi-service setup would use RS256 (asymmetric) so services can
# verify without knowing the signing key.
_ALGORITHM = "HS256"

# FastAPI security scheme — auto-generates the "Authorize" button in Swagger docs.
# In Spring terms, this is the SecurityFilterChain config for Bearer token auth.
_bearer_scheme = HTTPBearer()


def create_token(username: str, role: str) -> str:
    """Encode a JWT with username, role, and expiry.

    Args:
        username: the authenticated user's username (becomes the 'sub' claim).
        role: the authenticated user's role (custom claim).

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "exp": now + timedelta(hours=settings.jwt_expiry_hours),
    }
    # jwt.encode returns a str in PyJWT >=2.0 (older versions returned bytes).
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT, returning typed claims.

    Args:
        token: raw JWT string from the Authorization header.

    Returns:
        TokenPayload with sub, role, and exp.

    Raises:
        AuthenticationError: if the token is expired, tampered, or malformed.
    """
    try:
        # PyJWT verifies exp automatically and raises ExpiredSignatureError.
        raw = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
        return TokenPayload(**raw)
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired — please log in again.")
    except jwt.InvalidTokenError as e:
        # Catches all other JWT errors: bad signature, malformed, missing claims.
        raise AuthenticationError(f"Invalid token: {e}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> TokenPayload:
    """FastAPI dependency: extract and validate JWT from the Authorization header.

    Usage in endpoints:
        @router.post("/chat")
        async def chat(user: TokenPayload = Depends(get_current_user)):
            # user.role is the authenticated role — never trust client-supplied role

    This is the equivalent of Spring's @PreAuthorize + SecurityContext.
    FastAPI resolves the dependency chain: Request → HTTPBearer → get_current_user.
    The resolution is eager — all dependencies in the chain are resolved before
    the endpoint handler runs.

    Raises:
        AuthenticationError: if token is missing, expired, or invalid.
    """
    return decode_token(credentials.credentials)
