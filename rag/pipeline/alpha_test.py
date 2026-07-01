import logging
import time
import truststore

truststore.inject_into_ssl()

from common.log_utils import setup_logging
from rag.factory import EmbedderFactory
from rag.pipeline.chunker import chunk
from rag.pipeline.indexer import index
from rag.pipeline.parser import parse
from rag.pipeline.rerank import rerank_results
from rag.pipeline.report import write_results_md
from rag.pipeline.store import init_db
from rag.search.alpha_test import embed_query, run_search
from dotenv import load_dotenv

load_dotenv()

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ────────────────────────── Choosing the Embedder ───────────────────────────
# The Instance returned by `interactive_pick()` is a locked combination of
# (template, tokenizer, provider, distance metric). Everything downstream —
# chunking (tokenizer), indexing (provider + metric), search (metric),
# storage (model_key) — flows from it. To skip the interactive picker in
# code, use `EmbedderFactory.from_template(...)` or
# `EmbedderFactory.from_saved_instance(name)` instead.

instance = EmbedderFactory.interactive_pick()
logger.info("alpha_test instance=%s", instance.describe())

# ────────────────────────── Parsing the Document ────────────────────────────

file_path = "data/bcs.pdf"

fingerprint = parse(file_path)
logger.info("alpha_test parsed file=%s fingerprint=%s", file_path, fingerprint)

# ────────────────────────── Chunking the Document ───────────────────────────

init_db()
chunks = chunk(fingerprint, instance.count_tokens, instance.model_key)
logger.info("alpha_test built/loaded chunks=%d", len(chunks))

# ────────────────────────── Indexing the Document Chunks ─────────────────────

index(chunks, instance)

# ────────────────────────── Preparing the Query ──────────────────────────────


query = "Mike calls Jimmy for help, and Jimmy comes to the rescue"
#query = "Mike asks for a lawyer named McGill / Mike slides Jimmy McGill's business card / Jimmy arrives at the police station and Mike asks him to spill coffee on the detective"
start_time = time.perf_counter()
query_vector = embed_query(instance, query)
end_time = time.perf_counter()
logger.info("alpha_test embed_query time=%s", end_time - start_time)

# ────────────────────────── Running the Searches ─────────────────────────────

k= min(20, len(chunks)//4)

ann_time,   ann_results   = run_search("ANNSearcher",   query_vector, query, fingerprint, instance, k)
ivf_time,   ivf_results   = run_search("IVFSearcher",   query_vector, query, fingerprint, instance, k)
lsh_time,   lsh_results   = run_search("LSHSearcher",   query_vector, query, fingerprint, instance, k)
annoy_time, annoy_results = run_search("AnnoySearcher", query_vector, query, fingerprint, instance, k)
knn_time,   knn_results   = run_search("KnnSearcher",   query_vector, query, fingerprint, instance, k)
# logger.info("ANNSearcher time=%s",   ann_time)
# logger.info("IVFSearcher time=%s",   ivf_time)
# logger.info("LSHSearcher time=%s",   lsh_time)
# logger.info("AnnoySearcher time=%s", annoy_time)
# logger.info("KnnSearcher time=%s",   knn_time)

# ────────────────────────── Reranking the Top-K (per searcher) ───────────────
# No-op when `instance.reranker is None`; otherwise hydrates chunk text
# from SQLite and replaces each searcher's results with a cross-encoder
# reranked ordering. The search elapsed time stays as the bi-encoder
# search time — rerank time is logged separately.

ann_rr_time,   ann_results   = rerank_results("ANNSearcher",   ann_results,   query, fingerprint, instance, k)
ivf_rr_time,   ivf_results   = rerank_results("IVFSearcher",   ivf_results,   query, fingerprint, instance, k)
lsh_rr_time,   lsh_results   = rerank_results("LSHSearcher",   lsh_results,   query, fingerprint, instance, k)
annoy_rr_time, annoy_results = rerank_results("AnnoySearcher", annoy_results, query, fingerprint, instance, k)
knn_rr_time,   knn_results   = rerank_results("KnnSearcher",   knn_results,   query, fingerprint, instance, k)

# ────────────────────────── Writing the Markdown Report ──────────────────────

runs = [
    ("ANNSearcher",   ann_time,   ann_results),
    ("IVFSearcher",   ivf_time,   ivf_results),
    ("LSHSearcher",   lsh_time,   lsh_results),
    ("AnnoySearcher", annoy_time, annoy_results),
    ("KnnSearcher",   knn_time,   knn_results),
]
reranker_label = instance.reranker_id if instance.reranker is not None else None
report_path = write_results_md(
    query, fingerprint, instance.model_key, runs, k, reranker_label=reranker_label
)
logger.info("alpha_test report written path=%s", report_path)
