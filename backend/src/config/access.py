ROLE_COLLECTIONS: dict[str, list[str]] = {
    "doctor":            ["clinical", "nursing", "general"],
    "nurse":             ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician":        ["equipment", "general"],
    "admin":             ["clinical", "nursing", "billing", "equipment", "general"],
}

SQL_RAG_ROLES: set[str] = {"billing_executive", "admin"}


def get_collections_for_role(role: str) -> list[str]:
    """Return collections accessible to a given role.

    Raises:
        KeyError: if role is not in ROLE_COLLECTIONS. Caller wraps as RBACError.
    """
    return ROLE_COLLECTIONS[role]


def get_roles_for_collection(collection: str) -> list[str]:
    """Return roles that may access a given collection.

    Derived dynamically from ROLE_COLLECTIONS — never a separate hardcoded map.

    Args:
        collection: one of the five collection names.

    Returns:
        List of role strings that have this collection in their access list.

    Raises:
        ValueError: if no roles map to collection (misconfigured collection name).
    """
    roles = [role for role, colls in ROLE_COLLECTIONS.items() if collection in colls]
    if not roles:
        raise ValueError(f"No roles mapped to collection '{collection}' — check ROLE_COLLECTIONS.")
    return roles
