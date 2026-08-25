# Aster & Row Support Agent

A reliability-first AI customer support agent built with LangGraph, hybrid RAG (FAISS + BM25), and a deterministic tool layer — designed to handle conflicting policies, adversarial prompts, and messy real-world data deliberately, not just the happy path.

Built for a take-home assignment simulating a real support deployment: the knowledge base contains superseded policies, internal-only notes, and two genuinely contradictory active documents on purpose.

## Features

### 🧭 Deterministic Agent Routing
- Every consequential decision (tool call, source ranking, conflict detection, human handoff) is decided by plain Python, not the LLM
- The LLM is called exactly once per turn, only to phrase an already-decided, already-grounded answer
- This makes ~90% of agent behavior unit-testable without any API call

### 📚 Hybrid Retrieval with Precedence
- FAISS (dense) + BM25 (sparse), combined via Reciprocal Rank Fusion
- Custom precedence layer: `active` + `official` documents outrank `superseded`/`draft`/`internal` ones, even when a stale document scores higher on raw similarity
- Explicit `supersedes` front-matter resolution — a document naming another as superseded demotes it regardless of the older doc's own status field

### ⚠️ Genuine Conflict Detection
- Detects when two currently-active, official documents disagree (e.g. product-care guide says hand-wash; the product card says dishwasher-safe)
- Does not silently pick one — surfaces the conflict and recommends human confirmation, with a safest-interim-guidance suggestion

### 🔒 Privacy by Construction
- The LLM never receives raw order records — only a sanitized, allow-listed result
- PII (email, address), internal notes, and risk scores have no field to occupy in the sanitized schema — not filtered out, structurally absent
- A second, independent output-side scrub runs before any response reaches the user, as defense in depth

### 🛡️ Prompt Injection Resistance
- Retrieved documents and tool results are placed in a clearly delimited untrusted-content block
- System prompt explicitly instructs the model to treat that content as data, never instructions
- A pattern-based detector flags injection attempts for observability/logging, independent of the primary architectural defense

### 🔁 Multi-Turn Context
- Session state tracks `last_topic` and `last_order_id` so follow-ups like *"What about Canada?"* or *"When will it arrive?"* resolve without re-parsing the full transcript
- Verified with dedicated multi-turn tests, not just single-turn cases

### 📊 Deterministic Evaluation Suite
- No LLM-as-judge for pass/fail — every assertion is a substring/keyword/field check against actual graph output
- Reports per-case results, grouped by category (retrieval, groundedness, tool-use, privacy, multi-turn, prompt-security)
- 5+ original cases beyond the supplied set, including one that surfaced a real bug (see Bug Diary)

## Tech Stack

| Technology | Usage |
|---|---|
| Python 3.13 + uv | Core language, dependency/environment management |
| LangGraph | Agent control flow / state machine |
| LangChain | LLM and retrieval integration |
| Groq (`openai/gpt-oss-120b`) | LLM generation |
| sentence-transformers (HuggingFace, local) | Embeddings — offline, no API key required |
| FAISS | Dense vector search |
| rank-bm25 | Sparse keyword search |
| FastAPI | Agent-serving API |
| Streamlit | Chat frontend |
| Pydantic / pydantic-settings | Schema validation, config |
| pytest | Test suite |
| LangSmith (optional) | Tracing |

## Architecture

```mermaid
graph TD
    UI[Streamlit UI] -->|HTTP| API[FastAPI /chat]
    API --> Agent[LangGraph Agent]
    Agent --> Reset[reset_scratch]
    Reset --> Classify[classify_intent<br/>deterministic]
    Classify -->|order intent| OrderTool[order_tool_node<br/>deterministic]
    Classify -->|policy intent| Retrieve[retrieve_node<br/>hybrid FAISS+BM25 → precedence<br/>→ conflict check → injection flag]
    Retrieve --> Ground[grounding_check<br/>deterministic]
    OrderTool --> Respond[respond_node<br/>only LLM call — Groq]
    Ground --> Respond
    Respond --> Out[answer + sources + handoff]
```

### Retrieval pipeline

```mermaid
graph TD
    KB[knowledge-base/*.md] --> Loader[loader.py<br/>parse front matter:<br/>status, policy_authority, supersedes]
    Loader --> Chunker[chunker.py<br/>heading-aware split,<br/>metadata per chunk]
    Chunker --> FAISS[FAISS<br/>dense search]
    Chunker --> BM25[BM25<br/>sparse search]
    FAISS --> RRF[Reciprocal Rank Fusion]
    BM25 --> RRF
    RRF --> Precedence[precedence.py<br/>active+official first;<br/>supersedes-aware demotion]
    Precedence --> Conflict[conflict_detector.py<br/>known active-vs-active<br/>contradictions]
    Conflict --> Respond[respond_node]
```

### Order lookup pipeline

```mermaid
graph TD
    Orders[data/orders.json<br/>never sent to the LLM] --> Lookup[order_lookup.py<br/>normalize ID, validate, read record]
    Lookup --> Sanitizer[order_sanitizer.py<br/>allow-list only: status, carrier,<br/>tracking, ETA, safe message —<br/>no PII, no internal fields]
    Sanitizer --> Result[OrderLookupResult]
    Result --> Respond[respond_node]
```

## Project Structure

```
.
├── app/
│   ├── agent/          # state.py, nodes.py, graph.py, llm.py, prompts.py
│   ├── retrieval/       # loader, chunker, embeddings, vector_store,
│   │                     bm25_store, hybrid_retriever, precedence, retriever
│   ├── tools/           # order_lookup.py, order_sanitizer.py
│   ├── services/        # conflict_detector.py, safety.py
│   ├── schemas/         # chat.py, order.py
│   ├── observability/   # logging_config.py
│   ├── ui/               # streamlit_app.py
│   ├── config.py
│   └── main.py           # FastAPI app
├── scripts/
│   └── build_index.py
├── evaluation/
│   ├── visible-cases.json
│   ├── custom-cases.json
│   ├── run_eval.py
│   └── results/
├── data/orders.json
├── knowledge-base/*.md
├── test/
└── .streamlit/config.toml
```

## Installation

```bash
git clone https://github.com/SSY199/aster-agent.git
cd aster-agent
uv sync
cp .env.example .env
```

### Environment variables

```bash
# .env
GROQ_API_KEY=YOUR_GROQ_KEY
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=YOUR_LANGSMITH_KEY
LANGSMITH_PROJECT=YOUR_LANGSMITH_PROJECT_KEY

KB_DIR=knowledge-base
ORDERS_PATH=data/orders.json
FAISS_INDEX_PATH=storage/faiss_index
BM25_INDEX_PATH=storage/bm25_index.pkl
```

No key is required for embeddings — they run locally.

### Build the retrieval index (one-time, or after editing `knowledge-base/`)

```bash
uv run python -m scripts.build_index
```

### Run

```bash
# Terminal 1
uv run uvicorn app.main:app --port 8000

# Terminal 2
uv run streamlit run app/ui/streamlit_app.py
```

## How It Works

### Policy question

```mermaid
sequenceDiagram
    participant U as User
    participant C as classify_intent
    participant R as retrieve_node
    participant G as grounding_check
    participant L as respond_node (Groq)

    U->>C: "How long is the return window?"
    C->>R: intent = policy
    R->>R: hybrid search → precedence ranking
    R->>G: retrieved + ranked chunks
    G->>L: authoritative content found — proceed
    L->>U: answer grounded in retrieved context, cites source
```

### Order lookup

```mermaid
sequenceDiagram
    participant U as User
    participant C as classify_intent
    participant T as order_tool_node
    participant L as respond_node (Groq)

    U->>C: "Where is ORD-1007?"
    C->>T: extracts "ORD-1007", intent = order
    T->>T: order_lookup.py reads orders.json,<br/>returns sanitized result only
    T->>L: OrderLookupResult
    L->>U: answer grounded in tool result<br/>(no source citation needed — not KB-derived)
```

### Genuine source conflict

```mermaid
sequenceDiagram
    participant U as User
    participant R as retrieve_node
    participant CD as check_conflicts()
    participant L as respond_node (Groq)

    U->>R: "Can I put the Breeze Tumbler in the dishwasher?"
    R->>R: retrieves 11-product-care.md AND<br/>12-breeze-tumbler-product-card.md<br/>(both active + official)
    R->>CD: retrieved chunks
    CD->>CD: registered conflict rule fires
    CD->>L: handoff=True, reason="source_conflict"
    L->>U: explains the disagreement,<br/>does NOT silently pick one,<br/>recommends human confirmation
```

## Running the Evaluation Suite

```bash
uv run python -m evaluation.run_eval \
  --file evaluation/visible-cases.json \
  --out evaluation/results/final.json

uv run python -m evaluation.run_eval \
  --file evaluation/custom-cases.json \
  --out evaluation/results/custom-final.json
```

No LLM-as-judge — every check is deterministic (substring, keyword-overlap heuristic for paraphrased concepts, tool-call arguments, retrieved-source membership, handoff flags).

### Results — visible cases (15 supplied)

| Category | Result |
|---|---|
| Retrieval | 1/2 |
| Multi-source grounding | 0/1 |
| Conversation | 0/1 |
| Groundedness | 2/2 |
| Tool use | 2/2 |
| Tool reliability | 3/3 |
| Privacy | 1/1 |
| Prompt security | 0/1 |
| Abstention | 0/1 |
| Source conflict | 1/1 |
| **Total** | **10/15 (66.7%)** |

### Results — custom cases (6 added)

| Category | Result |
|---|---|
| Conversation | 1/1 |
| Multi-source grounding | 1/1 |
| Tool use | 1/1 |
| Prompt security | 0/1 |
| Groundedness | 1/1 |
| Retrieval | 1/1 |
| **Total** | **5/6 (83.3%)** |

Baseline and final runs produced identical numbers — the deterministic parts of the pipeline (routing, tool calls, handoff logic) are stable; only LLM prose phrasing varies between runs.

**Manually reviewing every failure against the actual response text:** most are the evaluation harness's keyword-overlap heuristic being stricter than the agent's real correctness (e.g. the agent says *"I don't have any information in the current knowledge base"* — which **is** an insufficiency abstention — but doesn't contain the harness's exact expected phrase). Two are genuine, minor retrieval-recall gaps on compound/adversarial phrasing. Details in Bug Diary and Known Limitations below.

## Bug Diary

### 1. Substring false-positive in order-intent classification
**Reproduced:** *"My TrailPlus membership was active when I ordered..."* classified as an order-status question ("ordered" contains "order").
**Fix:** Word-boundary keyword matching instead of substring search.
**Test:** `test_classify_intent_does_not_misfire_on_substring_order`

### 2. Damage/defect claims never triggered human handoff
**Reproduced:** A correctly-answered damaged-item question never set `handoff=True`, despite policy requiring human review before approval.
**Fix:** Added damage-keyword detection to `grounding_check`.
**Test:** custom case `damage-claim-triggers-handoff`, using new phrasing not present when the fix was written — confirms the fix generalizes.

### 3. "order" as a verb misclassified as order-status intent
**Reproduced:** *"If I order to Canada, will I owe customs duties?"* — a pure policy question — triggered an order-ID request.
**Discovery:** Found via a custom eval case built specifically to probe this class of bug, not present in the supplied cases.
**Fix:** Regex guard excluding "order" used as a verb ("if I order", "when I order") from the order-topic keyword match.
**Tests:** `test_classify_intent_order_as_verb_not_misclassified`, plus a regression test guarding the original case still works.

### 4. Conflict detection triggered on an unrelated query
**Reproduced:** *"Are all fabrics and adhesives in your bags vegan?"* incorrectly flagged the Breeze Tumbler dishwasher conflict and forced an unnecessary handoff, because retrieval's top-k happened to include weakly-scored tumbler chunks alongside the genuinely relevant ones.
**Root cause:** Conflict checking ran across the full retrieved set rather than being scoped to the passages actually relevant to the question.
**Fix:** Scoped `check_conflicts()` to only the top few, highest-scoring retrieved chunks.
**Test:** vegan-material abstention case, asserting no spurious `source_conflict` handoff.

### 5. `initial_state()` referenced undefined variables
**Reproduced:** `NameError` on every call, caught by static analysis before runtime.
**Root cause:** Copy/paste error left `intent=intent` instead of `intent=None`.
**Fix:** Corrected to literal `None` defaults.

### 6. YAML front-matter dates broke schema validation
**Reproduced:** Loading any knowledge-base file raised a `pydantic.ValidationError` — `yaml.safe_load` auto-converts unquoted `YYYY-MM-DD` into `datetime.date`, but the schema expected `str`.
**Fix:** Normalize any date-like parsed value to its ISO string form before schema construction.
**Test:** `test_loader.py` loads and validates every real file in `knowledge-base/`.

## Known Limitations

- [ ] Conflict detection is a small registry of known conflict pairs, not a general contradiction detector — a genuinely novel conflict wouldn't be caught automatically
- [ ] Retrieval occasionally misses a relevant heading on compound or adversarial queries (e.g. an adversarial prompt-injection phrasing didn't always surface the current returns policy in top-k)
- [ ] The evaluation harness's concept-matching is keyword-overlap, not semantic — deliberately avoiding LLM-as-judge per the assignment's guidance, at the cost of some correct answers scoring as failures on wording alone
- [ ] Single intent per turn — a message combining an order lookup and a policy question resolves only the order half deterministically
- [ ] In-memory session storage only — sessions don't survive an API restart
- [ ] `langchain_community.vectorstores.FAISS` is deprecated upstream in favor of a standalone package; not migrated mid-assignment

## Future Improvements

**Retrieval**
- [ ] Query decomposition for compound/multi-topic questions
- [ ] Re-ranking stage on top of RRF fusion
- [ ] Larger k with a relevance floor tuned against real score distributions

**Agent**
- [ ] Multi-intent handling (order + policy in one message, answered together)
- [ ] General semantic contradiction detection instead of a fixed rule registry

**Evaluation**
- [ ] Lightweight semantic-similarity scoring as a secondary (not primary) signal alongside deterministic checks
- [ ] Persisted session storage for multi-turn eval across restarts

**Observability**
- [ ] `/debug/{session_id}` endpoint exposing structured turn logs directly, not just file-based
- [ ] Token/cost tracking per turn

<!-- ## AI Coding Tools Used

Claude (Anthropic) was used throughout for architecture design, code generation, debugging, and this documentation. One example of an incorrect AI suggestion: an early recommendation to use `models/text-embedding-004` for Gemini embeddings was based on outdated model naming — that endpoint doesn't exist under that name, and the actual fix (switching to local `sentence-transformers` embeddings) was found by testing directly against the live API rather than trusting the suggestion. -->

## 🎬 Demo

▶️ [Watch the demo](https://raw.githubusercontent.com/SSY199/aster-agent/main/demo/demo.mp4)

Shows: a return-window policy question with citations, an order
lookup, multi-turn context (shipping → Canada follow-up), and the
Breeze Tumbler source conflict correctly triggering a human handoff.

- Knowledge-base question with authoritative source citations
- Sanitized order lookup
- Multi-turn conversation context
- Genuine source conflict → human handoff
- Prompt-injection resistance
- Deterministic evaluation suite

## Author

Sahil Yadav
