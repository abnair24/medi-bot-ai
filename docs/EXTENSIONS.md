# MediBot — Extensions Reference

> **Status update:** With the adoption of LangGraph, conversational memory (multi-turn
> chat) and human-in-the-loop (HITL SQL) are now **native features** built into the core
> pipeline in Phase 5 — not Phase 11 extensions. LangGraph's `SqliteSaver` checkpointer
> handles memory, and `interrupt()` handles HITL natively.
>
> This file is retained as **design reference** for the architectural rationale, golden
> case schemas, and eval track definitions that are now part of the core eval harness
> (EVAL_PLAN.md tracks 8–10).
>
> **What remains as future extensions:** additional graph features (parallel node
> execution, streaming responses, sub-graphs for complex multi-step reasoning).

## Extension Status

| # | Extension | Status |
|---|---|---|
| 1 | Conversational Memory (Multi-Turn Chat) | ✅ Done — Phase 5 (LangGraph SqliteSaver + rewrite_node) |
| 2 | Human-in-the-Loop (HITL) for SQL RAG | ✅ Done — Phase 5 (LangGraph interrupt() + /chat/resume) |
| 3 | Eval Tracks for Extensions | 🔲 Planned — Phase 8 |
| 4 | Agentic SQL Tool-Loop (Sub-Graph) | 🔲 Planned — Phase 12 |
| 5 | Distributed Tracing (Arize Phoenix) | 🔲 Planned — Phase 12 |

---

## Extension 1: Conversational Memory (Multi-Turn Chat) ✅ DONE

> **Implementation note:** Built in Phase 5 via LangGraph. `SqliteSaver` checkpointer
> persists conversation state per `thread_id`. `rewrite_node` handles query rewriting
> using prior turns. `should_rewrite` edge skips rewrite on first turn. `original_question`
> preserved in state so SQL path is not contaminated by rewrite context.
> See `backend/src/graph/nodes.py`, `backend/src/graph/edges.py`, `backend/src/services/rewrite_service.py`.

### 1.1 What it is

The current `/chat` endpoint is stateless — each request is independent. Conversational
memory lets the user ask follow-up questions that reference prior turns:

```
Turn 1: "What is the first-line drug for Type 2 Diabetes?"
Turn 2: "What's the max daily dose?"          ← refers to Metformin from Turn 1
Turn 3: "Any contraindications?"              ← still refers to Metformin
```

Without memory, Turn 2 goes to the retriever with "max daily dose" — no drug name, no
context, garbage retrieval.

### 1.2 Architectural design

**Conversation store:** a bounded, per-session sliding window of prior turns.

```
User → /chat (session_id + question)
            │
       ┌────┴────┐
       │ CONVERSATION STORE
       │   load last N turns for session_id
       │   (N = 5 turns, configurable)
       └────┬────┘
            │
       CONTEXT-AWARE QUERY REWRITE ──────── LLM call (new, #0)
            │   input: current question + prior turns
            │   output: self-contained rewritten query
            │   e.g., "max daily dose" → "maximum daily dose of Metformin"
            │
            ▼
       EXISTING PIPELINE (router → retrieval/SQL → generation)
            │   the rewritten query is what enters the pipeline
            │   everything downstream is unchanged
            │
            ▼
       STORE TURN
            │   save (question, answer) to conversation store
            │   trim to sliding window
```

**Key design decisions:**

1. **Query rewrite, not context injection.** Don't stuff all prior turns into the
   retrieval query or the generation prompt. Instead, use a lightweight LLM call to
   rewrite the ambiguous follow-up into a self-contained query. This keeps the retrieval
   and generation stages clean — they don't need to know about conversation history.

2. **Sliding window, not full history.** LLM context windows are finite. Cap at 5 turns
   (configurable). This is the same idea as log rotation in distributed tracing — you
   keep enough to correlate, not everything.

3. **Session-scoped, not user-scoped.** A session ends when the Streamlit tab closes or
   the JWT expires. No persistent conversation history across sessions.

4. **In-memory store for demo.** Use a simple dict keyed by `session_id`. For production,
   swap for Redis via the adapter pattern (a `ConversationStoreAdapter` behind an
   interface — same DI pattern as the rest of the app).

### 1.3 New/modified files

| File | Change | Type |
|---|---|---|
| `backend/src/models/schemas.py` | Add `session_id: str \| None = None` to `ChatRequest` | MODIFY |
| `backend/src/services/conversation_service.py` | ConversationStore: add_turn, get_history, sliding window | NEW |
| `backend/src/services/query_rewrite_service.py` | LLM-based query rewrite using prior turns | NEW |
| `backend/src/api/chat_router.py` | Wire conversation store + rewrite before pipeline | MODIFY |
| `docs/PROMPTS.md` | Add §8: query rewrite prompt | MODIFY |
| `frontend/components/chat.py` | Generate and send `session_id` per chat session | MODIFY |

### 1.4 Query rewrite prompt (for PROMPTS.md)

```
You are a query rewriter. Given a conversation history and the user's latest message,
rewrite the latest message as a fully self-contained question.

Rules:
- Resolve all pronouns ("it", "that", "they") using conversation context.
- Include relevant entity names, drug names, codes, or topics from prior turns.
- If the latest message is already self-contained, return it unchanged.
- Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest message: {question}

Rewritten question:
```

---

## Extension 2: Human-in-the-Loop (HITL) for SQL RAG ✅ DONE

> **Implementation note:** Built in Phase 5 via LangGraph. `interrupt()` in
> `sql_propose_node` suspends the graph and returns a `SQLProposal` to the API.
> `/chat/resume` endpoint accepts `ConfirmRequest` and calls `graph.ainvoke(Command(resume=...))`
> to continue execution. TTL enforced by comparing checkpoint `created_at` against
> `settings.hitl_proposal_ttl_seconds`. Toggle via `settings.sql_hitl_enabled`.
> See `backend/src/graph/nodes.py`, `backend/src/api/chat_router.py`, `backend/src/models/schemas.py`.

### 2.1 What it is

Currently, SQL RAG generates SQL and executes it immediately. Human-in-the-loop adds an
**approval step** between generation and execution: the user sees the proposed SQL,
approves or rejects, and only then does execution proceed.

In a medical/enterprise context, this is the difference between "the AI ran a query on
your database" and "the AI proposed a query — you approved it."

### 2.2 Architectural design

**Two-phase response pattern:**

```
Phase 1: Propose
    User → "how many claims are escalated?"
    Router → sql_rag
    SQL Gen → "SELECT COUNT(*) FROM claims WHERE status = 'escalated'"
    Response: {
        "proposed_sql": "SELECT COUNT(*) ...",
        "needs_confirmation": true,
        "confirmation_token": "abc123",    ← ties phase 2 to this proposal
        "answer": null
    }

Phase 2: Execute (after user approval)
    User → /chat/confirm {confirmation_token: "abc123", approved: true}
    Server → validate token, retrieve stored SQL
    Execute → run SQL against SQLite
    Narrate → LLM produces NL answer
    Response: {
        "answer": "There are 8 escalated claims.",
        "retrieval_type": "sql_rag",
        "executed_sql": "SELECT COUNT(*) ..."
    }
```

**Key design decisions:**

1. **Confirmation token, not re-sending SQL.** The client sends back a token, not the
   SQL string. This prevents a TOCTOU attack where the client modifies the SQL between
   proposal and execution. The server stores the proposed SQL and looks it up by token.
   (Think: compare-and-swap in distributed concurrency.)

2. **TTL on proposals.** Unconfirmed proposals expire after 5 minutes. This prevents
   stale queries from being executed later against a changed database state.

3. **Configurable enforcement.** HITL is enabled via `settings.sql_hitl_enabled: bool`.
   When disabled, SQL RAG behaves as before (auto-execute). This lets evals run without
   human interaction, and lets the demo toggle the feature.

4. **Hybrid RAG is not gated.** HITL only applies to SQL RAG. Hybrid RAG (document
   retrieval) is read-only by nature — it retrieves existing chunks, not executing
   arbitrary queries. No approval needed.

### 2.3 New/modified files

| File | Change | Type |
|---|---|---|
| `backend/src/models/schemas.py` | Add `SQLProposal`, `ConfirmRequest`, `ConfirmResponse` models | MODIFY |
| `backend/src/services/sql_rag_service.py` | Split into `propose()` and `execute_confirmed()` | MODIFY |
| `backend/src/services/proposal_store.py` | In-memory store: token → proposed SQL, with TTL | NEW |
| `backend/src/api/chat_router.py` | Add `/chat/confirm` endpoint; modify `/chat` for 2-phase | MODIFY |
| `backend/src/config/settings.py` | Add `sql_hitl_enabled: bool = False` | MODIFY |
| `frontend/components/chat.py` | Render SQL proposal with approve/reject buttons | MODIFY |

### 2.4 New Pydantic models (for schemas.py)

```python
class SQLProposal(BaseModel):
    """Returned when HITL is enabled and query is classified as SQL RAG."""
    proposed_sql: str
    confirmation_token: str
    needs_confirmation: Literal[True] = True
    expires_at: datetime  # UTC, server clock

class ConfirmRequest(BaseModel):
    confirmation_token: str
    approved: bool

class ConfirmResponse(BaseModel):
    answer: str | None  # None if rejected
    executed_sql: str | None
    retrieval_type: Literal["sql_rag"] = "sql_rag"
    role: str
    rejected: bool = False
```

---

## Extension 3: Eval Tracks for Extensions

### 3.1 Multi-turn retrieval quality (Track 8)

**What:** does query rewrite produce better retrieval than the raw follow-up question?

**Metric:** P@3 improvement over no-rewrite baseline.

**How:**
- Golden set: multi-turn conversation sequences (3–5 turns each). Each turn tagged with
  `expected_chunk_ids` for the rewritten query.
- Run two ways: (a) raw follow-up question → retriever, (b) rewritten question → retriever.
- Compare P@3 for both.
- Also check: does the rewrite **preserve the original intent**? A hallucinated rewrite
  that retrieves wrong chunks is worse than no rewrite.

**Golden cases (starter):**

| Turn sequence | Raw follow-up | Expected rewrite |
|---|---|---|
| T1: "First-line drug for Type 2 Diabetes?" → T2: "What's the max dose?" | "What's the max dose?" (fails) | "What is the maximum dose of Metformin for Type 2 Diabetes?" |
| T1: "ICU ventilator weaning protocol?" → T2: "What SpO2 threshold?" | "What SpO2 threshold?" (too vague) | "What SpO2 threshold is used in ICU ventilator weaning?" |
| T1: "Tell me about fault code F-05" → T2: "Which equipment?" | "Which equipment?" (garbage) | "Which equipment is associated with fault code F-05?" |
| T1: "Billing code for NSTEMI?" → T2: "What's the package rate?" | "What's the package rate?" (vague) | "What is the package rate for NSTEMI?" |
| T1: "Claim rejection rate?" → T2: "Break it down by department" | "Break it down by department" (needs SQL context) | "What is the claim rejection rate broken down by department?" |

**Tool:** plain Python (same as Track 1).

**Threshold:** rewrite P@3 ≥ baseline P@3 (no regression). Rewrite should improve P@3 by
≥ 0.10 on ambiguous follow-ups.

**Test file:** `backend/evals/test_multiturn.py`

**CI-gated?** Yes — threshold.

### 3.2 Query rewrite faithfulness (Track 9)

**What:** does the rewritten query preserve the user's intent, or does the LLM
hallucinate a different question?

**Metric:** intent preservation rate (binary per case).

**How:**
- For each multi-turn golden case, the rewritten query is compared against the expected
  rewrite.
- A DeepEval semantic-similarity check scores whether the rewrite conveys the same
  intent as the expected rewrite.
- Also: a **negative test** — inject a conversation history about Drug A, then ask about
  "the dose." The rewrite must reference Drug A, not hallucinate Drug B.

**Tool:** DeepEval (semantic similarity) + custom assertion.

**Threshold:** intent preservation ≥ 0.90.

**Test file:** `backend/evals/test_multiturn.py` (same file, separate test class).

**CI-gated?** Yes — threshold.

### 3.3 HITL SQL safety (Track 10)

**What:** when HITL is enabled, does the system correctly gate SQL execution behind
user approval? Does it reject expired/tampered tokens?

**Metric:** pass/fail assertions.

**How:**
- **Happy path:** propose → confirm → execute. Assert correct result.
- **Rejection path:** propose → reject. Assert no SQL executed, answer is None.
- **Expiry path:** propose → wait > TTL → confirm. Assert rejection with expiry message.
- **Tampered token:** send a random token to `/chat/confirm`. Assert 404/403.
- **Replay attack:** use a confirmed token again. Assert rejection (token consumed).
- **Auto-execute mode:** disable HITL via config. Assert SQL executes without proposal.

**Tool:** plain pytest (endpoint integration tests).

**Threshold:** all cases pass. Any failure → build fails.

**Test file:** `backend/evals/test_sql_hitl.py`

**CI-gated?** Yes — fail-closed.

### 3.4 Golden dataset additions (for backend/evals/golden/)

**New file: `backend/evals/golden/multiturn.json`**

Schema:
```json
{
  "conversations": [
    {
      "id": "mt-001",
      "turns": [
        {
          "turn": 1,
          "question": "What is the first-line drug for Type 2 Diabetes?",
          "expected_rewrite": null,
          "expected_chunk_ids": ["clinical_treatment_protocols_diabetes_s1"]
        },
        {
          "turn": 2,
          "question": "What's the max daily dose?",
          "expected_rewrite": "What is the maximum daily dose of Metformin for Type 2 Diabetes?",
          "expected_chunk_ids": ["clinical_drug_formulary_metformin"]
        }
      ],
      "description": "Pronoun resolution: 'the max dose' → Metformin from Turn 1"
    }
  ]
}
```

**New file: `backend/evals/golden/sql_hitl.json`**

Schema:
```json
{
  "cases": [
    {
      "id": "hitl-001",
      "question": "How many claims are escalated?",
      "role": "billing_executive",
      "test_type": "happy_path",
      "expected_sql_contains": "escalated",
      "expected_result": 8,
      "approval": true
    },
    {
      "id": "hitl-002",
      "question": "Total approved amount by department?",
      "role": "admin",
      "test_type": "rejection",
      "approval": false,
      "expected_answer": null
    },
    {
      "id": "hitl-003",
      "question": "Rejection rate?",
      "role": "billing_executive",
      "test_type": "token_expiry",
      "approval": true,
      "delay_seconds": 301,
      "expected_error": "proposal_expired"
    }
  ]
}
```

---

## Updated eval summary (including extensions)

| Track | What it measures | Tool | CI-gated? | Phase |
|---|---|---|---|---|
| 1. Retrieval A/B | P@3, MRR — dense vs hybrid vs +rerank | numpy | Yes — threshold | 8 |
| 2. RBAC red-team | leak rate across role × collection × attack | pytest | Yes — **zero** | 8 |
| 3. Numeric faithfulness | dose/code correct and from context | DeepEval | Yes — threshold | 8 |
| 4. SQL correctness | valid SQL + correct result + faithful NL | pytest | Yes — pass/fail | 8 |
| 5. Routing accuracy | confusion matrix (analytical vs knowledge) | Python | Yes — threshold | 8 |
| 6. Chunk quality | parent heading present, tables intact | pytest | Yes — pass/fail | 8 |
| 7. Judge calibration | judge agreement with human labels | Python | No — reported | 8 |
| **8. Multi-turn retrieval** | **rewrite → P@3 improvement over raw follow-up** | **numpy** | **Yes — threshold** | **11** |
| **9. Rewrite faithfulness** | **intent preservation after query rewrite** | **DeepEval** | **Yes — threshold** | **11** |
| **10. HITL SQL safety** | **token lifecycle: confirm/reject/expire/tamper/replay** | **pytest** | **Yes — fail-closed** | **11** |

---

## Build sequence for extensions

| Phase | What | Depends on | Deliverable |
|---|---|---|---|
| **11a** | Conversational memory + query rewrite | Phase 6 (API working) | Multi-turn chat works, rewrite logged |
| **11b** | HITL for SQL RAG | Phase 5 (SQL RAG working) | 2-phase SQL flow, approve/reject in UI |
| **11c** | Extension evals | Phases 11a + 11b | Golden sets, 3 new eval tracks, CI green |

---

## Production considerations for extensions

### Memory store scaling

In-memory dict works for single-instance demo. For multi-instance deployment:
- Swap `ConversationStore` for a Redis-backed adapter (same interface, different impl).
- Add TTL to conversations (e.g., 30 min inactive → evict).
- The adapter pattern in the existing architecture makes this a one-file swap.

### Query rewrite latency

The rewrite LLM call adds ~200–400ms per request (Groq latency). This is acceptable
for interactive chat but adds up. Mitigation:
- Skip rewrite when `session_id` is None or conversation history is empty (first turn).
- Use a smaller/faster model for rewrite than for generation (if available).

### HITL UX

The 2-phase flow changes the Streamlit chat UX. The chat component needs to:
- Display proposed SQL in a code block with syntax highlighting.
- Show approve/reject buttons inline.
- Handle the async round-trip (propose → wait → confirm).
- Show a clear "expired" state if the user takes too long.

---

## Extension 4: Agentic SQL Tool-Loop (Sub-Graph) 🔲 PLANNED — Phase 12

### 4.1 What it is

The current SQL RAG path is a **linear pipeline**: generate SQL → execute once →
narrate. It cannot handle:

- Multi-step queries ("show escalated claims; for each, show maintenance tickets from
  that department the same week" → two queries with a join)
- Self-correction ("0 rows returned — maybe the date filter is too narrow, broaden it")
- Format decisions based on result shape (1 row → prose; 20 rows → table)
- Follow-up refinements without a full new conversation turn

The agentic SQL tool-loop replaces the two linear nodes (`sql_propose_node`,
`sql_execute_node`) with a **sub-graph** where the LLM has `execute_sql` as a tool,
observes the result, and decides whether to run another query or produce a final answer.

This is the ReAct pattern (Reason → Act → Observe → repeat) applied to SQL.

### 4.2 Architecture

```
Supervisor node
    │
    └── routes to → SQL Agent Sub-Graph
                        │
                        ┌─────────────────────────────────┐
                        │  sql_agent_state:               │
                        │    question, role, history      │
                        │    tool_calls: []               │
                        │    iterations: 0                │
                        │    answer: None                 │
                        └─────────────────────────────────┘
                        │
                        ▼
                   [sql_think_node]
                   LLM (70B) reasons:
                   - what SQL do I need?
                   - do I have enough data?
                   - is the result a table or a number?
                        │
               ┌────────┴────────┐
               │                 │
          needs_tool           has_answer
               │                 │
               ▼                 ▼
        [sql_act_node]        [sql_format_node]
        execute_sql()         LLM (8B) formats:
        append to             - table (markdown)
        tool_calls            - prose (sentence)
               │                 │
               └────────┬────────┘
                        │
                   loop back or END
                   (max 5 iterations)
```

### 4.3 Key design decisions

**1. Tool definition — execute_sql as a LangGraph tool**

```python
from langchain_core.tools import tool

@tool
def execute_sql(query: str) -> dict:
    """Execute a read-only SQL query against the hospital database.
    Returns rows as a list of dicts with column names as keys.
    patient_name and patient_id are automatically stripped from results.
    """
    result = sql_adapter.execute_query(query)
    return _strip_phi(result).model_dump()
```

The tool docstring is what the LLM reads to understand what it can do. PHI strip
happens inside the tool — the LLM never sees patient_name or patient_id.

**2. Max iterations guard**

```python
MAX_SQL_ITERATIONS = 5

def should_continue(state: SQLAgentState) -> str:
    if state["iterations"] >= MAX_SQL_ITERATIONS:
        return "force_answer"   # LLM ran too many queries — summarise what you have
    if state.get("answer"):
        return "end"
    return "think"
```

Prevents runaway loops. If the agent reaches 5 iterations without an answer,
a force-answer node summarises what was retrieved and returns it.

**3. Task-based model split within the agent**

```python
# sql_think_node — reasoning, SQL generation
think_response = await llm.generate(prompt, model="groq/llama-3.3-70b-versatile")

# sql_format_node — template-like formatting of known result
format_response = await llm.generate(prompt, model="groq/llama-3.1-8b-instant")
```

The think node needs 70B for schema reasoning and multi-step planning. The format
node is deciding "is this a table or a sentence?" — 8B is more than sufficient.

**4. HITL redesign for agent loop**

The current HITL design assumes one SQL per turn. An agent loop may run 3 queries.
Two options:

- **Approve-first:** Show the agent's initial SQL plan (first query) and approve before
  any execution. The agent runs subsequent queries without approval (they are derived,
  not user-initiated).
- **Approve-all:** Pause at each `execute_sql` tool call. More control, worse UX.

Recommended: **approve-first**. The first query establishes intent. Derived follow-up
queries (e.g., joining a second table) are mechanical consequences the agent reasoned
to, not new user intents.

**5. Format decision is the agent's responsibility**

The current narration prompt hardcodes "write natural language." In the agent:

```
Tool result: [{"department": "cardiology", "count": 23}, {"department": "nephrology"...}]

LLM sees this and decides:
  - 7 rows, 2 columns → markdown table
  - 1 row, 1 column  → one sentence
  - 0 rows           → "no matching records found"
```

No prompt rule needed. The agent reasons about what it sees.

### 4.4 How it replaces the current linear path

**Current (Phase 5):**
```
graph.add_node("sql_rag", make_sql_propose_node(sql_rag_svc))
graph.add_node("sql_execute", make_sql_execute_node(sql_rag_svc))
graph.add_conditional_edges("sql_rag", route_after_propose, {...})
graph.add_edge("sql_execute", END)
```

**Phase 12 (agent sub-graph):**
```python
sql_agent = build_sql_agent(llm=llm, sql=sql, settings=settings)

graph.add_node("sql_agent", sql_agent)   # sub-graph replaces two nodes
graph.add_edge("sql_agent", END)
# route_after_propose and sql_execute nodes are removed
```

The supervisor still routes to "sql_agent" via `route_after_router`. The internal
loop is encapsulated in the sub-graph — the outer graph sees one node.

### 4.5 New/modified files

| File | Change | Type |
|---|---|---|
| `backend/src/graph/sql_agent.py` | SQL agent sub-graph: think → act → format loop | NEW |
| `backend/src/graph/sql_agent_state.py` | `SQLAgentState` TypedDict: tool_calls, iterations, answer | NEW |
| `backend/src/adapters/sql_adapter.py` | Add `execute_query()` as a LangGraph-compatible tool | MODIFY |
| `backend/src/graph/builder.py` | Wire `sql_agent` sub-graph, remove `sql_propose`/`sql_execute` nodes | MODIFY |
| `backend/src/graph/nodes.py` | Remove `make_sql_propose_node`, `make_sql_execute_node` | MODIFY |
| `backend/src/graph/edges.py` | Remove `route_after_propose` | MODIFY |
| `backend/src/services/sql_rag_service.py` | Extract tool-callable `execute_query()` from `execute_and_narrate()` | MODIFY |
| `backend/src/config/settings.py` | Add `sql_agent_max_iterations: int = 5` | MODIFY |
| `backend/evals/test_sql_agent.py` | Agent loop evals: multi-step, format, self-correction | NEW |

### 4.6 Eval additions (Track 11)

| Test | What it verifies |
|---|---|
| Multi-step query | Agent runs 2 queries and combines results correctly |
| Self-correction | Agent gets 0 rows, broadens filter, gets result |
| Table format | Multi-row result is rendered as markdown table |
| Prose format | Single aggregate is rendered as a sentence |
| Max iterations | Agent forced-answers at iteration limit, does not loop forever |
| RBAC inside tool | `execute_sql` tool rejects queries from unauthorised roles |
| PHI in tool output | `patient_name`, `patient_id` never appear in agent's final answer |

### 4.7 Production considerations

**Rate limits:** Each tool call = one SQL execution + potentially one more LLM call.
5 iterations = up to 10 LLM calls per request. Use task-based routing to keep
think-node on 70B and format-node on 8B. Monitor Groq rate limit headers.

**Read-only enforcement:** The `execute_sql` tool must validate the query is a SELECT
before running it. Any DDL (CREATE, DROP, ALTER) or DML (INSERT, UPDATE, DELETE) is
rejected with an error returned to the agent.

**Observability:** Each tool call is a discrete event. Log `(iteration, sql, row_count,
latency_ms)` per tool call. This makes multi-step agent traces debuggable without
logging query content (PHI risk).

**LiteLLM + Bedrock alignment:** Task-based model selection inside the agent is the
natural fit for LiteLLM proxy routing. `think_node` requests `medibot-sql-70b`,
`format_node` requests `medibot-narrate-8b`. LiteLLM config maps these to Groq or
Bedrock depending on environment.

---

## Extension 5: Distributed Tracing with Arize Phoenix 🔲 PLANNED — Phase 12

### 5.1 What it is

The pipeline already has **PHI-safe request-level observability** (built alongside
Phase 8/10):

- **Structured JSON logs** correlated by `request_id` (in every stage event, and in the
  `X-Request-ID` response header so a user can quote it when reporting an issue).
- **Per-stage latency** — `latency_ms` on `route_decision_detailed`, `retrieval`,
  `rerank`, `generation`, `sql_generated`, `sql_executed`; total `duration_ms` on the
  `http_request` event.
- **Token/cost accounting** — a per-request token meter (`src/utils/token_meter.py`,
  a mutable counter in a `ContextVar` so it survives LangGraph's child-task execution)
  totals Groq `usage.total_tokens`; surfaced in the `X-LLM-Tokens` header and
  `http_request` log.

What's missing is a **visual trace**: a per-request waterfall showing each node/LLM/
retrieval span with timing, inputs (redacted), and scores — plus aggregate dashboards
(p50/p95 latency per stage, token spend over time, error rates). `request_id` gives log
correlation, not a span tree.

### 5.2 Why Arize Phoenix

- Open-source, runs locally (Docker), no vendor lock-in — fits the self-hosted stack.
- Native **OpenInference/OpenTelemetry** instrumentation for LangGraph, LangChain, and
  the OpenAI/Groq clients — auto-captures spans with minimal code.
- Built for LLM apps: shows retrieval spans with chunk scores, LLM spans with token
  counts and latency, and lets you drill from an aggregate p95 spike to the exact trace.

### 5.3 Design (fits the existing three-layer rule)

- Add a `src/observability/tracing.py` module that configures an OTel tracer + Phoenix
  exporter at app startup (behind `settings.tracing_enabled`, default False — no-op when
  off, so nothing changes for local/demo).
- Instrument at the **boundary**, not in business logic: auto-instrument the Groq client
  and wrap `graph.ainvoke` in a root span carrying `request_id` as a span attribute, so
  logs and traces share the same correlation ID.
- **PHI stays out of spans**: reuse the existing discipline — span attributes carry IDs,
  scores, latencies, token counts, route/specificity — never query or answer text.
  A redaction processor drops any span attribute matching the query/answer keys.
- Reuse the token meter and `latency_ms` values already computed — Phoenix visualizes
  what's already measured; it doesn't replace the PHI-safe logs (which remain the audit
  record).

### 5.4 New/modified files

| File | Change | Type |
|---|---|---|
| `src/observability/tracing.py` | OTel tracer + Phoenix exporter, PHI-redaction span processor, `init_tracing()` | NEW |
| `src/api/app.py` | Call `init_tracing()` in lifespan when `settings.tracing_enabled` | MODIFY |
| `src/config/settings.py` | Add `tracing_enabled: bool = False`, `phoenix_endpoint: str` | MODIFY |
| `docker-compose.yml` | Add optional `phoenix` service (behind a compose `profile`) | MODIFY |
| `requirements.txt` | `arize-phoenix`, `openinference-instrumentation-*`, `opentelemetry-sdk` | MODIFY |

### 5.5 What it unlocks

- Per-request span waterfall: router → (rewrite) → retrieve → rerank → generate, each
  with timing + scores, one click from a slow request.
- Aggregate dashboards: p50/p95 latency per stage, token spend and cost trend (directly
  actionable against the Groq daily-token cap), error/rate-limit frequency.
- A shared correlation ID (`request_id`) across logs, the `X-Request-ID` header, and
  traces — so a user-reported issue maps to one trace and one log stream.
