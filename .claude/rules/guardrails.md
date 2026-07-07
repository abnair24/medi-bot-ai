# MediBot is a retrieval tool, not a medical advisor.
# These rules apply to every file in this repo.

## Medical safety — hard boundaries

- Never generate code that produces diagnostic conclusions ("the patient has X").
- Never generate code that prescribes or recommends treatment ("you should take X").
- All clinical/nursing answers must include a disclaimer appended by the generation
  service (deterministic, not left to the LLM).
- The LLM prompt must always contain "answer ONLY from the provided context."
- If retrieval returns no relevant chunks, the response must say so — never fabricate
  an answer or citation.
- Drug doses, lab values, and clinical thresholds must be quoted exactly as they
  appear in the retrieved context. Never round, interpolate, or convert units.

## RBAC — enforcement layer

- The access_roles metadata filter must be applied on the Qdrant query itself
  (pre-filter), never as a post-retrieval filter in application code.
- A user must never see chunks from a collection outside their role's access list,
  regardless of query phrasing or prompt injection attempts.
- SQL RAG is gated to billing_executive and admin only. Check role before executing.

## Response format

- Every answer must include source citations (source_document + section_title).
- Every clinical/nursing answer must end with the medical disclaimer.
- Every drug-related answer must end with the prescribing disclaimer.
- RBAC refusals must name the user's role and list their accessible collections.
