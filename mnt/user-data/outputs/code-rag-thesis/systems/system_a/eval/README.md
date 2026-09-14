# Leg 1 question set

`leg1_questions.json` holds 45 questions (15 per repo) covering the three categories
arXiv:2601.08773 evaluates: repository semantics, multi-hop architectural tracing, and
system-level reasoning.

The paper does not publish its exact question wording, so this file is a
**reconstructed comparable set**. State that plainly in the report (section 6, "known
deviations"); do not claim question-level parity with the paper.

Schema per entry:

```json
{
  "id": "shopizer-01",
  "repo": "shopizer",
  "category": "multi_hop_trace",
  "question": "Which service handles order payment confirmation, and which modules does it call?",
  "reference_answer": "Written by you from the source at the pinned commit.",
  "keywords": ["OrderService", "PaymentModule"],
  "gold_files": ["sm-core/src/main/java/.../OrderServiceImpl.java"]
}
```

- `category`: `repository_semantics` | `multi_hop_trace` | `system_reasoning`
- `keywords`: drive the automatic rubric (`shared/metrics.qa_rubric_score`)
- `gold_files`: optional, useful for retrieval sanity checks

Entries whose `question` still starts with `TODO` are counted and warned about at run time.
The same file is reused verbatim by Systems B and C - do not fork it per system.
