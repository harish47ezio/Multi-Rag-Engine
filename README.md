# Multi RAG Engine

> A pluggable RAG harness. Any LLM. Any embedder. Any retrieval strategy. Any distance metric.
> One contract, four retrieval brains, zero coupling.

Most RAG projects are built around a single retrieval recipe. **Multi RAG Engine** flips
that: it's a research-grade harness where the LLM, the embedder, the searcher, and the
distance metric are all swappable behind clean abstract contracts. The retrieval *brain*
how chunks are actually selected and enriched — is itself a pluggable module.

---

## Why it exists

Every retrieval strategy (BM25, dense KNN, HNSW, IVF, LSH, GraphRAG, HippoRAG, Agentic…)
shines on a different shape of problem. There is no universal winner. This project lets you
hold **everything else constant** — same document, same embedder, same LLM — and swap only
the retrieval brain to see what actually changes.

That makes it useful as:

- A **reference implementation** of a novel two-pass iterative retrieval algorithm
  (described below).
- A **harness** for trying any LLM × embedder × searcher × metric combination.
- A **benchmarking surface** for retrieval research.

---

## The novel idea: Two-Pass Iterative Retrieval

The default retrieval brain (`mode = iterative`) is a custom two-pass scheme:

```
query
  │
  ▼
[Pass 1]  embed(query)  →  search()  →  top-k chunks
  │
  ▼
[Pass 2]  for each chunk in top-k:
              embed(chunk)  →  search()  →  supporting-k chunks
  │
  ▼
[Merge + dedupe]  pass-1 ∪ pass-2  →  enriched context
  │
  ▼
[LLM]  answer(query, enriched_context)
```

The LLM never touches the loop. It is called exactly once, at the end, with a
context window that has already been *associatively* expanded — every top-k hit
pulls in its own neighbours, so single-hop weak matches still surface their
supporting evidence.

Compared to plain top-k:

| Plain top-k                          | Two-pass iterative                        |
| ------------------------------------ | ----------------------------------------- |
| One vector lookup                    | k+1 vector lookups, all parallelisable    |
| Misses supporting context            | Each hit pulls in its own neighbourhood   |
| LLM does the stitching at answer time| Stitching happens before the LLM sees it  |
| Cheap, fast, brittle on multi-hop    | Slightly slower, much stronger recall     |

If no document is provided, the engine scrapes the web for the topic and treats the
scraped corpus as the source document — same pipeline, same brain.

---

## Architecture

Four orthogonal layers, each defined by an abstract contract:

```
┌────────────────────────────────────────────────────────────────┐
│  LLM Harness         BaseLLMAdapter        complete()          │
│  ─ ollama (local + cloud) Done                                 │
│  ─ openai, anthropic, gemini, hf  (factory slot ready)         │
├────────────────────────────────────────────────────────────────┤
│  Embedder            BaseEmbedder          embed() / dimension │
│  ─ ollama (local + cloud) Done                                 │
├────────────────────────────────────────────────────────────────┤
│  Searcher            BaseSearcher          index() / search()  │
│  ─ BM25Searcher        (lexical, rank_bm25)                    │
│  ─ KNNSearcher         (exact, numpy)                          │
│  ─ ANNSearcher         (HNSW, hnswlib)                         │
│  ─ AnnoySearcher       (random projection forest)              │
│  ─ IVFSearcher         (inverted file, FAISS)                  │
│  ─ LSHSearcher         (random hyperplanes / SimHash)          │
├────────────────────────────────────────────────────────────────┤
│  Distance Metric     BaseDistanceMetric    index_matrix() ...  │
│  ─ Cosine     (similarity)                                     │
│  ─ Dot        (similarity)                                     │
│  ─ Euclidean  (distance)                                       │
└────────────────────────────────────────────────────────────────┘
```

The retrieval *brain* sits on top of all four and is itself swappable:

```
mode = [ iterative | hippo | graph | agentic ]
```

Same document store, same harness, same I/O — different brain.

### Why the layers are split this way

- **Provider client vs role adapter.** `providers/ollama_client.py` holds connection state
  (base URL, API key, headers). The role adapters (`llm/ollama_adapter.py`,
  `rag/embedder/ollama_adapter.py`) bind a *model* to that client for a specific role.
  One client, many roles, zero duplication.
- **Embedder vs tokenizer.** The transport (provider → model → vector) is `BaseEmbedder`;
  token counting (chunker concern) is `BaseTokenizer`. They live in separate packages
  (`rag/embedder/` and `rag/tokenizer/`) because the same model can be served by N
  providers with no tokenizer change, and a single provider can serve N models with N
  different tokenizers. The `EmbeddingInstance` (built by `rag/factory/`) composes one of each.
- **Four building blocks, one mother instance.** Embedding, search, reranker, and LLM are
  four *independent* registry building blocks — each with its own template (what's
  available) and named instance (a locked pick). A `MotherInstance` ties one of each
  together (reranker + LLM optional) and is the single object the pipeline runs on. You
  pick a mother instance at startup; everything downstream flows from it.
- **Metric is behaviour, not data.** Each ANN backend (hnswlib, FAISS, Annoy) speaks its
  own metric vocabulary. The `MetricKind` enum is the conceptual identity; each searcher
  owns its own mapping from `MetricKind` → backend constant. Mismatches throw
  `UnsupportedMetricError` at construction, not at search time.
- **Persistence is per-searcher.** Every searcher implements `save(path)` / `load(path)`
  — native binary for graph/index files, JSON sidecar for Python-side state (texts,
  hyperparameters, metric kind). Metric is verified on load.
- **Document identity is the fingerprint.** A SHA-256 of the input file is computed
  once by the parser and threaded through every downstream stage. Parsed markdown,
  chunks, vectors, every searcher index, and every per-query report are all keyed by
  the fingerprint, so reruns are cache-hits and per-document artifacts never collide.

---

## Retrieval matrix

| Searcher          | Backend          | Best for                                | Cosine | Dot | Euclid |
| ----------------- | ---------------- | --------------------------------------- | :----: | :-: | :----: |
| `BM25Searcher`    | rank_bm25        | exact terms, IDs, names                 |   —    |  —  |   —    |
| `KNNSearcher`     | numpy            | small corpora, ground-truth recall      |   yes  |  yes|   yes  |
| `ANNSearcher`     | hnswlib (HNSW)   | large corpora, high recall, in-memory   |   yes  |  yes|   yes  |
| `AnnoySearcher`   | Annoy            | static corpus, low RAM, immutable index |   yes  |  yes|   yes  |
| `IVFSearcher`     | FAISS IVF        | very large corpora, tunable recall      |   yes  |  yes|   yes  |
| `LSHSearcher`     | numpy SimHash    | streaming, zero training budget         |   yes  |  yes|   yes  |

Same `BaseSearcher` interface for every row. The retriever never branches on
backend — it just calls `search(query_vector, query_text, top_k)`.

---

## Runtime pipeline

`parse → chunk → index → search → rerank → synthesize → report` — every stage is keyed by
the document's SHA-256 fingerprint, so reruns are cache-hits and per-document artifacts
never collide. `rerank` and `synthesize` are no-ops when the mother instance carries no
reranker / LLM.

### Document fingerprinting

Every document is identified by `sha256(file_bytes)` (hex). The fingerprint is computed
once by the parser (`fingerprint/hash.py`) and threaded through every downstream stage:

- `parse(path)` → returns the fingerprint and writes `output/<fingerprint>.md` (the
  intermediate, page-marked markdown). Re-running the parser on the same file is a
  cache hit — no re-conversion.
- `chunk(fingerprint, count_tokens, model_key)` → reads `output/<fingerprint>.md`,
  persists chunks to SQLite keyed by `(fingerprint, model_key, chunk_index)`.
  Different embedding models produce different token boundaries, so chunks are
  scoped per model. Re-running with the same model returns the cached chunks.
- `index(chunks, embedding)` → writes embeddings and every searcher index to
  `storage/<fingerprint>/<model_key>/`. Per-searcher files are skipped if already
  present. The metric used for every searcher comes from `embedding.metric` —
  never hard-coded. (`embedding` is the mother instance's `EmbeddingInstance`.)
- `run_search(name, ...)` → loads the per-(fingerprint, model_key) index for one
  searcher and returns `(elapsed, [(chunk_index, score), ...])`. Only the strategies
  listed in the mother's search instance run.
- `rerank_results(name, results, query, fingerprint, embedding, reranker, top_n)` →
  hydrates chunk text and replaces a searcher's bi-encoder ordering with the
  cross-encoder reranker's. No-op (returns results unchanged) when `reranker` is None.
- `synthesize_answer(results, query, fingerprint, embedding, llm, top_n?)` → stitches the
  top results' text into a grounded prompt and asks the LLM to answer. Returns the
  `LLMResponse`, or None when `llm` is None.
- `write_results_md(query, fingerprint, model_key, runs, top_k, reranker_label?, llm_answer?)`
  → hydrates `chunk_index`es back to full text + page ranges via the SQLite store, writes
  `result/<fingerprint>_<model_key>_<timestamp>.md`. When `llm_answer` is present it is
  rendered as an `## LLM Answer` section at the top.

Net effect: change the document or the model → everything re-runs. Don't change
either → every stage is an O(1) cache hit.

### Chunking strategy

Token-aware, paragraph- and sentence-aware. Sizing is in **tokens** (using the
tokenizer attached to the active `EmbeddingInstance` via `embedding.count_tokens`), not characters.

Primary split — ladder, prefer larger semantic units:

1. Whole tail fits the budget → emit it.
2. Largest paragraph boundary that fits.
3. Largest sentence boundary that fits (sentences segmented with `pysbd`).
4. Hard token cut (binary search on character offset).

Overlap — ladder, walked from chunk end backward:

1. Last full paragraph (if it fits the overlap budget).
2. Last 4, then 3, then 2, then 1 sentences.

The parser injects `<!-- page: N -->` markers between PDF pages. The chunker strips
them while building a `page → cleaned-offset → raw-offset` map, so every chunk
carries both a flattened `text` (what the embedder sees) and
`start_page` / `end_page` / `page_char_start` / `page_char_end` (used by the report
writer to re-slice the raw markdown for high-fidelity snippets).

Default config (`ChunkConfig`): 512 tokens per chunk, 15% overlap.

### SQLite chunk store

Chunks live in `storage/chunks.db` (table `chunks`, PK `(fingerprint, model_key, chunk_index)`):

| Column                            | Purpose                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `fingerprint`, `model_key`, `chunk_index` | Composite primary key. `model_key` scopes chunks per embedding model because different tokenizers produce different boundaries. |
| `text`                            | Flattened chunk text (what the embedder sees).                                                                |
| `start_page`, `end_page`          | Page span this chunk covers.                                                                                  |
| `source`                          | Path to `output/<fingerprint>.md`.                                                                            |
| `page_char_start`, `page_char_end`| Page-relative offsets, used by the report writer to slice the raw markdown for high-fidelity snippets.        |
| `overlap_source`                  | Which ladder rung produced this chunk's overlap (`paragraph`, `sent:4`, …) or `NULL`.                         |

Read APIs (`rag/pipeline/store.py`):

- `init_db()` — create the table (idempotent, call once before chunking).
  If a legacy table without `model_key` is detected it is dropped and recreated.
- `get_chunk(fp, model_key, idx)` — single chunk.
- `get_multi_chunks(fp, model_key, [idx, ...])` — bulk hydrate (used by the report writer).
- `get_all_chunks(fp, model_key)` — every chunk for a (document, model), in `chunk_index` order.

### Per-query reports

`rag/pipeline/report.write_results_md(query, fingerprint, model_key, runs, top_k, reranker_label?, llm_answer?)`
writes `result/<fingerprint>_<model_key>_<timestamp>.md` containing:

- Header (query, fingerprint, top-k, timestamp).
- Optional `## LLM Answer` section (when `llm_answer` is passed).
- Summary table: searcher → elapsed seconds.
- Per-searcher section: each ranked hit shows `chunk_index`, score (annotated as
  cross-encoder score when `reranker_label` is set), page range, source path, and the
  **raw** snippet sliced out of `output/<fingerprint>.md` (preserving paragraphs and
  dialogue line breaks — not the flattened chunk text).

This is the harness's evaluation surface: same document, same query, every searcher
side by side, one diff away from each other.

---

## Project layout

```
Multi RAG Engine/
├── common/
│   └── log_utils.py              # setup_logging(), preview(), count_chars()
├── data/                         # input PDFs (gitignored)
├── fingerprint/
│   └── hash.py                   # sha256(file) → document identity
├── output/                       # parser writes output/<fingerprint>.md
├── result/                       # per-query markdown reports
├── registry.yaml                 # v2: embedding/reranker/llm templates + named instances + mother instances
├── storage/                      # SQLite + per-(fingerprint, model_key) index binaries (gitignored)
│   ├── chunks.db                 # SQLite chunk store
│   └── <fingerprint>/<model_key>/  # vectors.npy + index_{hnsw,annoy,ivf,lsh}
├── providers/
│   └── ollama_client.py          # HTTP client (local + cloud), no model bound
├── llm/                          # Completion role
│   ├── base.py                   # BaseLLMAdapter, LLMResponse
│   ├── factory.py                # LLMFactory (provider registry)
│   └── ollama_adapter.py
└── rag/
    ├── embedder/                 # Embedding role — transport (text → vector)
    │   ├── base_embedder.py      # BaseEmbedder contract (embed, dimension, recommended_metric)
    │   └── ollama_adapter.py
    ├── tokenizer/                # Token counting — orthogonal to provider
    │   ├── base_tokenizer.py     # BaseTokenizer contract (count_tokens)
    │   └── hf_tokenizer.py       # HuggingFace tokenizers backend
    ├── factory/                  # Build runtime instances from the registry
    │   ├── factory.py            # EmbeddingFactory / RerankerFactory / LLMFactory / MotherFactory
    │   └── instance.py           # EmbeddingInstance / RerankerInstance / LLMInstance / SearchInstance / MotherInstance
    ├── registry/                 # Templates + named instances + mother instances + CLI
    │   ├── schema.py             # EmbeddingTemplate / RerankerTemplate / LLMTemplate / *InstanceSpec / MotherInstanceSpec
    │   ├── loader.py             # YAML load + atomic save (schema_version 2)
    │   ├── validator.py          # validate_tokenizer / validate_provider / validate_reranker / per-category picks
    │   ├── picker.py             # run_picker(): pick or assemble a mother instance
    │   ├── wizard.py             # register-{embedding,reranker,llm}-template + save-{instance,mother} flows
    │   ├── cli.py                # list / register-template / register-reranker / register-llm / save-mother / validate / refresh
    │   └── __main__.py           # python -m rag.registry CLI entrypoint
    ├── pipeline/
    │   ├── models.py             # Chunk, ChunkConfig dataclasses
    │   ├── parser.py             # PDF → output/<fp>.md  (TXT/MD: in progress)
    │   ├── chunker.py            # token + paragraph + sentence ladder
    │   ├── store.py              # SQLite chunk store
    │   ├── indexer.py            # embed → build & persist all indices
    │   ├── rerank.py             # rerank_results(): cross-encoder re-ordering of a searcher's top-k
    │   ├── synthesize.py         # synthesize_answer(): grounded LLM answer over retrieved chunks
    │   └── report.py             # write_results_md(): per-query markdown report
    └── search/
        ├── base_search.py        # BaseSearcher contract
        ├── _validation.py        # Shared index() validators
        ├── bm25.py
        ├── distance_metrics/
        │   ├── base_distance_metric.py
        │   ├── cosine.py · dot.py · euclidean.py
        ├── knn/knn_searcher.py
        └── ann/
            ├── ann.py            # HNSW
            ├── annoy.py          # Annoy
            ├── ivf.py            # FAISS IVF
            └── lsh.py            # SimHash LSH
```

---

## Quickstart

### 1. Install

```bash
git clone <this-repo>
cd "Multi RAG Engine"
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure

Copy the env template and fill it in:

```bash
cp .env.example .env
```

`.env` holds one variable today — `OLLAMA_API_KEY`. When it's needed:

| Setup                                              | Needs `OLLAMA_API_KEY`?         |
| -------------------------------------------------- | :-----------------------------: |
| Local Ollama (`base_url=http://localhost:11434`)   | No — just run `ollama serve`    |
| Ollama Cloud (`base_url=https://ollama.com`)       | Yes                             |

Get a cloud key at <https://ollama.com/settings/keys>. The key is loaded via
`python-dotenv` (`from dotenv import load_dotenv; load_dotenv()`) — done for you
in every `alpha_test.py`. On corporate networks add
`import truststore; truststore.inject_into_ssl()` before the first request so
the system trust store is used for TLS.

### 3. Run the LLM harness

```python
from dotenv import load_dotenv; load_dotenv()
import truststore; truststore.inject_into_ssl()
from llm import LLMFactory

llm = LLMFactory.create({
    "provider": "ollama",
    "model": "llama3.1:8b",
    "base_url": "http://localhost:11434",   # or "https://ollama.com" for cloud
})

print(llm.is_available())
print(llm.complete("Summarise transformers in one sentence.").text)
```

### 4. Pick (or assemble) a mother instance

The registry has four independent building blocks, each with a **template** (what's
available) and a **named instance** (a locked pick):

| Block     | Template lists…                          | Instance locks…                         |
| --------- | ---------------------------------------- | --------------------------------------- |
| Embedding | model → metric, dimension, tokenizers, providers | one tokenizer + provider (+ metric override) |
| Reranker  | reranker model → providers               | one provider                            |
| LLM       | LLM model → providers                    | one provider                            |
| Search    | catalogue of searcher strategies         | a subset of strategies to run           |

A **mother instance** references one instance of each (reranker + LLM optional). It's the
single object the pipeline runs on. The repo ships `registry.yaml` (schema v2) with seed
templates and example instances (`default`, `full-stack`). Inspect and extend via the CLI:

```bash
python -m rag.registry list                    # every template + instance + mother instance
python -m rag.registry register-template       # wizard: define a new embedding template
python -m rag.registry register-reranker       # wizard: define a new reranker template
python -m rag.registry register-llm            # wizard: define a new LLM template
python -m rag.registry save-mother             # assemble + save a mother instance
python -m rag.registry validate qwen3-embedding-8b   # re-check one template (any kind)
python -m rag.registry refresh                 # re-check every template
```

In code:

```python
from rag.factory import MotherFactory

# Re-hydrate a saved mother instance (embedding + search + optional reranker + LLM):
mother = MotherFactory.from_saved("default")

# Interactive — pick or assemble one (used by every alpha_test.py):
mother = MotherFactory.interactive_pick()

embedding = mother.embedding      # EmbeddingInstance: embed / count_tokens / metric / model_key
strategies = mother.search.strategies   # e.g. ["ANNSearcher", "KNNSearcher"]
reranker = mother.reranker        # RerankerInstance or None
llm = mother.llm                  # LLMInstance or None
```

Each building block also has its own factory (`EmbeddingFactory`, `RerankerFactory`,
`LLMFactory`) with `from_template(...)` / `from_saved_instance(name)` when you want just
one piece:

```python
from rag.factory import EmbeddingFactory

embedding = EmbeddingFactory.from_template(
    model_key="qwen3-embedding-8b",
    tokenizer_id="hf-qwen3-8b",    # optional; defaults to first in template
    provider_id="ollama-local",    # optional; defaults to first in template
    # metric_override="dot",       # optional; logs a warning when used
)
```

### 5. Index a document

```python
from rag.factory import EmbeddingFactory
from rag.pipeline.chunker import chunk
from rag.pipeline.indexer import index
from rag.pipeline.parser  import parse
from rag.pipeline.store   import init_db

embedding = EmbeddingFactory.from_template(
    model_key="qwen3-embedding-8b",
    provider_id="ollama-local",
)

init_db()                                                   # one-time: create chunks table
fingerprint = parse("data/your_doc.pdf")                    # sha256(file); writes output/<fp>.md
chunks      = chunk(fingerprint, embedding.count_tokens,    # token-aware, persists to SQLite
                   embedding.model_key)
index(chunks, embedding)                                    # vectors.npy + HNSW + Annoy + IVF + LSH
                                                            # → storage/<fingerprint>/<model_key>/
```

All three calls are cache-hits on the second run for the same (file, model).

### 6. Search

```python
from rag.pipeline.store import get_multi_chunks
from rag.search.ann.ann import ANNSearcher

# Metric comes from the embedding instance — no hardcoded CosineDistanceMetric() anywhere.
searcher = ANNSearcher(metric=embedding.metric)
searcher.load(f"storage/{fingerprint}/{embedding.model_key}/index_hnsw.bin")

q_vec  = embedding.embed(["What is the architecture of a transformer?"])[0]
hits   = searcher.search(q_vec, "transformer architecture", top_k=5)   # [(chunk_index, score), ...]
chunks = get_multi_chunks(fingerprint, embedding.model_key, [idx for idx, _ in hits])

for (idx, score), c in zip(hits, chunks):
    print(f"[{score:.3f}] pages {c.start_page}-{c.end_page}: {c.text[:120]}…")
```

For the full harness, see `rag/pipeline/alpha_test.py` — it picks a mother instance,
runs the selected subset of searchers on the same query, reranks (if a reranker is
attached), synthesises an LLM answer (if an LLM is attached), and writes
`result/<fingerprint>_<model_key>_<timestamp>.md` via `write_results_md(...)`.

---

## Learn by example — `alpha_test.py`

Every module ships a small runnable script that exercises it end-to-end. Read these
before reading the abstractions — they're the fastest way to see how the harness
fits together.

| File                              | What it shows |
| --------------------------------- | ------------- |
| `llm/alpha_test.py`               | LLM harness — Ollama Cloud completion (needs `OLLAMA_API_KEY`). |
| `rag/embedder/alpha_test.py`      | Embedder harness — pick a mother instance, embed a sentence, print the dimension. |
| `rag/search/runner.py`            | Per-searcher loaders + `run_search()` helper used by the pipeline. |
| `rag/pipeline/alpha_test.py`      | **Start here.** Full end-to-end: pick a mother instance → parse PDF → chunk → embed → build all indices → run the selected searchers → rerank → synthesise LLM answer → write `result/<fingerprint>_<timestamp>.md`. |

`rag/pipeline/alpha_test.py` is the most representative — it touches the parser,
chunker, embedder, SQLite store, every searcher, and the report writer in one file.
If you only read one example, read that one.

Every `alpha_test.py` starts with `setup_logging("INFO")` from `common/log_utils.py`,
so you get the same structured log line format across all four scripts.

---

## Extending

### Add a new embedding model

1. Run `python -m rag.registry register-template` and answer the prompts.
2. The wizard validates the tokenizer (HF repo loads?) and each provider
   (reachable? serves the model?). Failures are saved too — fix them later
   and re-run `python -m rag.registry validate <model_key>` to flip status.
3. Use it from code via `EmbeddingFactory.from_template(model_key="...")`, or
   assemble it into a mother instance from the interactive menu.

### Add a reranker or LLM to the registry

1. Run `python -m rag.registry register-reranker` (or `register-llm`) and answer
   the prompts — a reranker/LLM template is a model with one or more providers.
2. The wizard validates each provider the same way (reachable? serves the model?).
3. Reference it by building a named reranker/LLM instance (via `save-mother` or the
   "assemble a new mother instance" flow), then attach it to a mother instance so
   the pipeline's rerank / synthesize stages activate.

### Add a new tokenizer kind (e.g. tiktoken, an API tokenizer)

1. Implement a subclass of `BaseTokenizer` in `rag/tokenizer/<your>Tokenizer.py`.
2. Add a branch in `rag/factory/factory._build_tokenizer()` for the new
   `kind` value (`"tiktoken"`, …).
3. Add a validation branch in `rag/registry/validator.validate_tokenizer`
   (construct your subclass — "constructible == valid").
4. Existing templates can now reference the new kind in their `tokenizers:` map.

### Add a new embedding provider kind (e.g. HuggingFace Inference, OpenAI)

1. Implement a new adapter mirroring `OllamaAdapter` (subclass `BaseEmbedder`,
   take a `metric_kind` in the constructor). Tokenizer concerns stay outside —
   providers never see tokenizers.
2. Add a branch in `rag/factory/factory._build_embedder()` for the new
   `kind` value (`"hf_api"`, `"openai"`, …).
3. Add a validation branch in `rag/registry/validator.validate_provider`.
4. Existing templates can now reference the new kind in their `providers:` map.

### Add a new LLM provider

1. Implement `BaseLLMAdapter` in `llm/<your_provider>Adapter.py`.
2. Register it in `llm/factory.py`:

   ```python
   if provider == "openai":
       return OpenAIAdapter(...)
   ```

The retriever code never changes.

### Add a new searcher

1. Implement `BaseSearcher` (`index`, `search`, `save`, `load`).
2. Map `MetricKind` → your backend's metric vocabulary.
3. Convert raw distance/similarity into a unified score (higher = better).
4. Wire it into `rag/pipeline/indexer.py`.

### Add a new distance metric

1. Implement `BaseDistanceMetric` (`index_matrix`, `search_matrix`,
   `index_query`, `metric_kind`).
2. Add a new `MetricKind` enum value.
3. Each ANN searcher decides whether to support it; unsupported pairings raise
   `UnsupportedMetricError` at construction.

---

## Status

| Component                                         | State |
| ------------------------------------------------- | :---: |
| LLM harness — Ollama (local + cloud)              |  Done |
| LLM harness — OpenAI / Anthropic / Gemini / HF    |   IP  | *factory slot ready, adapter deferred* |
| Embedder harness — Ollama                         |  Done |
| Registry — embedding / reranker / llm / search templates + instances | Done |
| Mother instance (composes one of each, picked at startup) | Done |
| Registry CLI (`python -m rag.registry ...`)       |  Done |
| Document fingerprint (SHA-256, cache key)         |  Done |
| Document parser — PDF (via `opendataloader-pdf`)  |  Done |
| Document parser — TXT / MD                        |   IP  | *fingerprint computed, intermediate markdown not yet written* |
| Token-aware paragraph+sentence chunker            |  Done |
| SQLite chunk store + bulk hydrate                 |  Done |
| Multi-index builder (HNSW + Annoy + IVF + LSH)    |  Done |
| Per-(fingerprint, model_key) storage scoping      |  Done |
| Searchers — BM25, KNN, HNSW, Annoy, IVF, LSH      |  Done |
| Distance metrics — Cosine, Dot, Euclidean         |  Done |
| Cross-encoder reranker stage (Ollama + HF)        |  Done |
| LLM answer synthesis over retrieved chunks        |  Done |
| Per-query markdown report writer (+ LLM answer)   |  Done |
| Centralised logging (`common/log_utils.py`)       |  Done |
| Mode `iterative` — two-pass retriever             |   IP  | *primitives Done, orchestration in progress* |
| Web-scrape fallback (no-doc mode)                 |   FS  |
| Mode `hippo` — HippoRAG associative recall        |   FS  |
| Mode `graph` — GraphRAG entity KG                 |   FS  |
| Mode `agentic` — retriever-as-tool, LLM loop      |   FS  |
| Mode switcher (`mode = …` config)                 |   FS  |

Done
IP: in progress 
FS: roadmap Future Scope

---

## Roadmap

- **Step 1.5 — Two-pass orchestrator.** Wrap the existing primitives into a single
  `IterativeRetriever` that runs pass-1, re-queries on each hit, merges, dedupes, and
  hands off to `llm.complete`.
- **Step 1.6 — Web-scrape fallback.** When no document is provided, scrape the topic and
  feed the scraped pages through the same parse → chunk → index pipeline.
- **Step 2 — HippoRAG mode.** Build associative node-edge map at index time; retrieval
  traverses associations like memory recall.
- **Step 3 — GraphRAG mode.** Extract entities and relationships at index time; retrieval
  walks the knowledge graph.
- **Step 4 — Agentic mode.** Wrap the retriever as a callable tool; let the LLM decide
  when to retrieve, what to search, and when to stop.
- **Step 5 — Mode switcher.** Single `mode = [iterative | hippo | graph | agentic]`
  config; same store, same harness, different brain.

---

## License

TBD — pick MIT / Apache-2.0 / etc. before sharing publicly.
