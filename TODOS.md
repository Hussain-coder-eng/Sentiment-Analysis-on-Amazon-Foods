# TODOS

## TODO-1: Review-trained RoBERTa model

**What:** Replace `cardiffnlp/twitter-roberta-base-sentiment` (Twitter-trained) with a model fine-tuned on product reviews — e.g. `nlptown/bert-base-multilingual-uncased-sentiment` or an Amazon-specific fine-tune on HF.

**Why:** The Twitter model is the weakest ML credibility point. An interviewer who asks "why use a Twitter sentiment model on Amazon reviews?" has a valid criticism. A review-trained model makes the disagreement visualization more defensible and scientifically meaningful.

**Pros:** Stronger ML credibility. More accurate disagreement signals. Better interview talking point ("I evaluated domain-specific models and found X").

**Cons:** Must re-run preprocessing to regenerate demo-data.json. Requires verifying the new model's HF Inference API response shape. May change disagreement patterns in the demo.

**Context:** Current model chosen for familiarity and benchmarking. A Phase 1 acceptable shortcut. Phase 2: swap model, regenerate demo data, compare disagreement distributions.

**Depends on:** Nothing. Independent.

---

## TODO-2: Async job/polling architecture for /api/analyze

**What:** Replace synchronous pipeline (Firecrawl + HF in one 10s serverless function) with async pattern: POST /api/analyze returns a job ID immediately, frontend polls GET /api/status/:id every 2s until complete.

**Why:** The synchronous approach is fragile on Vercel free tier even with Promise.all. Cold HF start + slow Firecrawl pages can still timeout. Async jobs remove the timeout constraint entirely and demonstrate async architecture knowledge.

**Pros:** Removes Vercel timeout risk permanently. Shows async/polling patterns (valuable interview signal). Enables longer Firecrawl runs (more reviews = better scatter plot).

**Cons:** ~4 hours additional implementation (human) / ~20min CC. Requires KV as job state store. More complex frontend (polling loop with abort). Overkill for Phase 1 portfolio.

**Context:** Phase 1 uses synchronous pipeline which is borderline with the 10s limit. Promise.all mitigates the Firecrawl bottleneck. If timeouts persist in production, this is the correct architectural fix.

**Depends on:** Phase 1 shipped. KV already integrated.
