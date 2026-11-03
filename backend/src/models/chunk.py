from pydantic import BaseModel, Field, model_validator
from pydantic.config import ConfigDict

from src.config.access import get_roles_for_collection
from src.models.schemas import Collection, ChunkType


class ChunkMetadata(BaseModel):
    """A validated chunk ready to be stored in Qdrant.

    Built by ingestion_service after Docling parsing and HierarchicalChunker.
    Stored as the Qdrant payload alongside dense and sparse vectors.
    """

    text: str = Field(min_length=1)
    source_document: str = Field(min_length=1)
    collection: Collection
    access_roles: list[str] = Field(min_length=1)
    section_title: str = Field(min_length=1)
    chunk_type: ChunkType

    @model_validator(mode="after")
    def validate_access_roles_match_collection(self) -> "ChunkMetadata":
        """Ensure access_roles is a subset of the roles allowed for this collection.

        Catches mismatches like collection='clinical' with access_roles=['nurse'],
        which would cause nurses to receive clinical chunks at query time.
        """
        allowed = set(get_roles_for_collection(self.collection))
        given = set(self.access_roles)
        invalid = given - allowed
        if invalid:
            raise ValueError(
                f"Roles {invalid} are not permitted for collection "
                f"'{self.collection}'. Allowed: {allowed}"
            )
        return self


class ChunkResult(BaseModel):
    """A chunk returned from Qdrant hybrid search, with its relevance score.

    access_roles is included from the Qdrant payload for audit logging.
    The RBAC pre-filter has already enforced access — these roles are metadata,
    not a second gate.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    score: float
    source_document: str = Field(min_length=1)
    collection: Collection
    access_roles: list[str] = Field(min_length=1)
    section_title: str = Field(min_length=1)
    chunk_type: ChunkType

    @model_validator(mode="after")
    def validate_access_roles_match_collection(self) -> "ChunkResult":
        """Ensure access_roles in the returned chunk is consistent with its collection.

        Guards against payload corruption in Qdrant — if a chunk comes back with
        mismatched roles, it's a data integrity error, not an RBAC enforcement issue.
        """
        allowed = set(get_roles_for_collection(self.collection))
        given = set(self.access_roles)
        invalid = given - allowed
        if invalid:
            raise ValueError(
                f"Returned chunk has invalid access_roles {invalid} for "
                f"collection '{self.collection}' — payload may be corrupted."
            )
        return self
