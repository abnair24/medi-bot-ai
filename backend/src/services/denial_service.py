"""Deterministic RBAC denial verbalizer.

Replaces the LLM-generated RBAC refusal in GenerationService with a structured,
template-driven denial message. Three reasons this is better than an LLM call:

1. Zero LLM token cost per denied request — at scale, RBAC probes and accidental
   out-of-scope queries are common; paying per-refusal is wasteful.

2. Auditable and loggable — the denial is a Pydantic model. Its fields (role,
   accessible_collections, denied_for) are safe to log as operational metadata.
   An LLM-generated refusal string is not: we'd have to log it or not, and if
   we do, we risk echoing the user's question back through the log.

3. Consistent surface — templated language makes denials predictable. Clinicians
   learn to expect a specific format; variable LLM wording causes confusion.

4. No hallucination risk — a template cannot accidentally hint at what is inside
   a restricted collection. An LLM occasionally does (guardrails.md §RBAC).

DenialReason is a Pydantic model so callers can serialize it, log its fields
individually, or render it as a string for the user. The `message` property
produces the user-visible text; callers use `.dict()` for structured logging.
"""

from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.config.access import ROLE_COLLECTIONS


class DenialReason(BaseModel):
    """Structured RBAC denial — serializable, loggable, and renderable.

    Fields are safe to log individually (no PHI, no answer text, no question text).
    The user-visible message is produced by the `message` property so the template
    lives in one place.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    accessible_collections: list[str]
    suggestion: str = (
        "If you need information outside your current access level, please contact "
        "your department head or a staff member with admin access."
    )

    @property
    def message(self) -> str:
        """User-visible denial text — deterministic, no LLM."""
        role_display = self.role.replace("_", " ").title()
        collections_display = ", ".join(
            c.replace("_", " ").title() for c in self.accessible_collections
        )
        return (
            f"As a {role_display}, your access is limited to the following "
            f"collections: {collections_display}. "
            f"The information requested is not available through your current access level. "
            f"{self.suggestion}"
        )


def verbalize_denial(role: str) -> DenialReason:
    """Build a DenialReason for the given role.

    Args:
        role: authenticated user role from the JWT payload.

    Returns:
        DenialReason with the role's accessible collections and a templated message.
        If role is not in ROLE_COLLECTIONS (should not happen — validated at auth time),
        returns a generic denial with an empty collections list.
    """
    accessible = ROLE_COLLECTIONS.get(role, [])
    return DenialReason(role=role, accessible_collections=accessible)
