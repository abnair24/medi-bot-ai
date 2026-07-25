"""Auth router — POST /login for JWT token issuance.

No authentication required on this endpoint (you're *getting* a token, not using one).
Credentials are checked against the hardcoded DEMO_USERS registry.
"""

from fastapi import APIRouter, status

from src.models.auth import DEMO_USERS, LoginRequest, LoginResponse
from src.security.jwt import create_token
from src.utils.exceptions import AuthenticationError
from src.utils.logger import logger

# prefix="" because /login sits at the API root, not under /auth.
# Tags group endpoints in Swagger UI — purely cosmetic but helpful for navigation.
router = APIRouter(tags=["auth"])


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate and receive a JWT",
)
async def login(body: LoginRequest) -> LoginResponse:
    """Validate credentials against the demo user registry and return a JWT.

    The JWT contains the user's role — all downstream /chat requests derive
    their RBAC context from this token, never from client-supplied data.

    Raises:
        AuthenticationError: if username not found or password doesn't match.
    """
    user = DEMO_USERS.get(body.username)

    if user is None or user.password != body.password:
        # Log the attempt (username is operational data, not PHI).
        # Never log the password — even a failed one.
        logger.warning(
            "login_failed",
            extra={"username": body.username, "reason": "invalid_credentials"},
        )
        raise AuthenticationError("Invalid username or password.")

    token = create_token(username=user.username, role=user.role)

    logger.info(
        "login_success",
        extra={"username": user.username, "role": user.role},
    )

    return LoginResponse(
        access_token=token,
        role=user.role,
        username=user.username,
    )
