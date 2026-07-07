# Code quality and engineering standards for all code in this repo.
# Full details in CODING_STANDARDS.md — this is the enforced summary.

## No hardcoded values

- All configuration comes from pydantic-settings (src/config/settings.py).
- Model names, top-k values, thresholds, ports, paths — everything is in Settings.
- Magic strings are forbidden. Use Literal types and typed constants.
- Role-collection access map is defined once in src/config/access.py.
- .env.example is committed; .env is gitignored.

## Design principles

- Single Responsibility: one file, one concern. Services don't do HTTP or SQL directly.
- Dependency Injection: services receive adapters, never construct them internally.
- Interface Segregation: adapters expose only what the service needs, not the full SDK.
- Open/Closed: swap implementations via new adapter files, not by modifying services.

## Pydantic everywhere

- Every API request/response uses a Pydantic model.
- Every chunk entering Qdrant is validated via ChunkMetadata.
- Every LLM structured output uses a Pydantic model.
- Every golden-set record is validated on load.
- Never pass unvalidated dicts between services.

## Error handling

- Never swallow exceptions silently (no bare `except: pass`).
- Use custom exceptions (RetrievalError, RBACError, SQLGenerationError, RoutingError).
- Map exceptions to HTTP responses in one place (app.py exception handlers).

## Adapter pattern

- External deps accessed only through adapters in src/adapters/.
- Service code imports adapters, never vendor SDKs directly.
- Adapters are injected so tests can substitute fakes.

## Naming and structure

- Files: snake_case. Classes: PascalCase. Functions: snake_case. Constants: UPPER_SNAKE.
- Explicit imports only (no `from x import *`).
- No circular imports.
- Every public function and class has a docstring.

## Eval rules

- Deterministic evals use plain pytest + numpy. No DeepEval for RBAC/SQL/routing.
- Judge-based evals (faithfulness, correctness) use DeepEval.
- Use a different model as judge than as generator.
- Eval thresholds imported from shared config, never hardcoded in test files.
- RBAC eval runs first in CI, fails fast. Any leak = build fails.

## Testing

- Unit tests mock adapters; eval tests use real adapters.
- Test names describe behavior: `test_nurse_cannot_retrieve_billing_chunks`.
- Use pytest fixtures for DI, not setup/teardown.
