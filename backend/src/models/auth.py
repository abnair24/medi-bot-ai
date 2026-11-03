"""Auth domain models and demo user registry.

Passwords are plain-text — acceptable for a portfolio demo (ARCHITECTURE.md §6).
Production would use bcrypt via passlib + a real user store.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict

from src.models.schemas import Role


# --- Domain models (not HTTP-facing) ---

class User(BaseModel):
    """A user in the demo registry. Looked up during /login."""

    model_config = ConfigDict(frozen=True)

    username: str
    password: str  # plain-text, demo only
    role: Role


class TokenPayload(BaseModel):
    """Claims decoded from a validated JWT.

    Maps 1:1 to the JWT payload: {"sub": username, "role": role, "exp": expiry}.
    Constructed in jwt.py after decode — never by application code directly.
    """

    model_config = ConfigDict(frozen=True)

    sub: str = Field(description="Username (JWT 'subject' claim)")
    role: Role
    exp: datetime

    @field_validator("exp", mode="before")
    @classmethod
    def parse_exp(cls, v: int | datetime) -> datetime:
        """Coerce PyJWT's int Unix timestamp to a timezone-aware datetime.

        PyJWT returns exp as an int. Pydantic v2 lax mode converts int→datetime
        implicitly, but enabling strict mode later would silently break that.
        Explicit coercion here is immune to mode changes.
        """
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v


# --- HTTP request/response models ---

class LoginRequest(BaseModel):
    """POST /login request body."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class LoginResponse(BaseModel):
    """POST /login response body."""

    model_config = ConfigDict(frozen=True)

    access_token: str
    token_type: str = "bearer"
    role: Role
    username: str


# --- Demo user registry (ARCHITECTURE.md §6) ---
# dict[username, User] — O(1) lookup by username.
# In Java terms, this is a static final Map<String, User> in a @Configuration class.

DEMO_USERS: dict[str, User] = {
    "dr.mehta": User(username="dr.mehta", password="doctor123", role="doctor"),
    "nurse.priya": User(username="nurse.priya", password="nurse123", role="nurse"),
    "billing.ravi": User(
        username="billing.ravi", password="billing123", role="billing_executive"
    ),
    "tech.anand": User(username="tech.anand", password="tech123", role="technician"),
    "admin.sys": User(username="admin.sys", password="admin123", role="admin"),
}
