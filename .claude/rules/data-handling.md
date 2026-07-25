# PHI and data handling rules for all code in this repo.

## Logging — what must NEVER be logged

- Never log the user's raw query text (may contain patient details).
- Never log the LLM's full response text (may echo patient details).
- Never log the full prompt sent to the LLM (contains the query).
- Never log patient_name or patient_id from SQL query results.
- Never log JWT tokens or credentials.

## Logging — what you SHOULD log

- Request ID, user role, route decision, chunk IDs + scores, latency, RBAC block status.
- These are operational metadata, not PHI.

## Code patterns to enforce

- Use structured logging with explicit field names — never f-string a query into a log.
- In sql_rag_service: strip patient_name and patient_id from results before passing to
  the narration LLM.
- Eval logs go to backend/evals/results/ (gitignored) and use only synthetic golden-set queries.
- Never write user queries to disk, database, or any persistent storage.
- Never include real patient data in test fixtures — use synthetic data only.

## .gitignore requirements

- backend/evals/results/ must be gitignored.
- .env must be gitignored.
- Any file containing API keys must be gitignored.
- qdrant_data/ (Docker volume) must be gitignored.
