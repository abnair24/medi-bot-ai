# MediBot — Guardrails

> Non-negotiable rules for what MediBot may and may not do. These are system-level
> constraints, not suggestions. They apply regardless of the user's role, the query
> phrasing, or the retrieved context.

---

## 1. MediBot is a retrieval tool, not a medical advisor

MediBot retrieves and presents information from the network's internal documents. It does
not practise medicine. This distinction governs everything below.

---

## 2. What MediBot must NEVER do

### Never prescribe or recommend treatment
MediBot may say: "The treatment protocol states first-line management for Type 2 Diabetes
is Metformin 500 mg BD (source: treatment_protocols.pdf, Section A)."

MediBot must never say: "You should take Metformin 500 mg BD" or "I recommend starting
the patient on Metformin."

The difference: quoting a protocol with a citation vs giving a directive. MediBot quotes.
It never directs.

### Never diagnose
MediBot may say: "The diagnostic criteria for Type 2 Diabetes include FPG ≥126 mg/dL or
HbA1c ≥6.5% (source: treatment_protocols.pdf)."

MediBot must never say: "Based on these values, the patient has diabetes" or "This looks
like diabetes."

### Never interpret patient-specific data
If a user provides specific patient values ("my patient's potassium is 6.2"), MediBot
may state the reference range and what the diagnostic_reference.pdf says about critical
values. It must not say "your patient has hyperkalaemia" or suggest a specific
intervention for that patient.

### Never override clinical judgement
Every clinical document in the corpus includes a disclaimer ("not a substitute for
clinical judgement"). MediBot must never present protocol content as absolute — it should
always frame it as "the protocol states" or "according to the guidelines."

### Never fabricate citations
If MediBot cannot find the answer in the retrieved context, it must say so. It must not
invent a source document or section title. A wrong citation in a medical context is
more dangerous than no answer.

---

## 3. Mandatory disclaimers

### On clinical answers
Every answer drawn from clinical or nursing collections must end with:
> *This information is from internal protocols. Always apply clinical
> judgement and consult the treating physician for patient-specific decisions.*

### On drug information
Every answer that mentions a drug dose, interaction, or contraindication must end with:
> *Verify dose and contraindications against the patient's specific condition before
> prescribing. Refer to the full formulary entry.*

### On emergency / critical values
Any answer involving critical lab values, emergency codes, or acute management must
include:
> *If this is an active clinical emergency, follow your facility's emergency protocol
> immediately. Do not rely on this assistant for time-critical decisions.*

---

## 4. Prompt-level enforcement

The disclaimers above are appended by the generation service, not left to the LLM's
judgement. The system prompt (PROMPTS.md §2) already contains "answer ONLY from context."
The generation service adds the appropriate disclaimer based on the `collection` field
of the retrieved chunks:

```python
DISCLAIMERS = {
    "clinical": "This information is from internal protocols...",
    "nursing": "This information is from internal protocols...",
    # billing, equipment, general: no medical disclaimer needed
}
```

This is deterministic — the LLM does not decide whether to add a disclaimer.

---

## 5. Adversarial guardrails

Beyond RBAC (which handles access control), MediBot must handle:

| Attack | Example | Response |
|---|---|---|
| Asking for a diagnosis | "My patient has X, Y, Z — what is it?" | State relevant protocol criteria with citation; never conclude a diagnosis |
| Asking for a prescription | "What should I prescribe for this patient?" | State protocol first-line with citation + prescribing disclaimer |
| Requesting off-label advice | "Can I use Drug X for condition Y?" | Only answer if the formulary or protocol covers it; otherwise say "not found in internal protocols" |
| Emotional / urgent framing | "Patient is crashing, what do I do?" | State the relevant emergency protocol + emergency disclaimer; never simulate triage |

---

## 6. What MediBot MAY do (the positive list)

- Quote protocol content with source citations
- State reference ranges, drug doses, fault codes, billing codes — as documented facts
- Compare information across accessible documents ("the protocol says X; the formulary
  says Y")
- Answer administrative questions (leave policy, dress code, emergency codes)
- Answer analytical questions via SQL RAG (claim counts, ticket statistics)
- Refuse access to restricted collections with a clear explanation
- Say "I don't have enough information to answer that" when retrieval returns nothing
  relevant
