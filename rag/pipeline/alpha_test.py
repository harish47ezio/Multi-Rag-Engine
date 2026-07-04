import logging
import time

from common.bootstrap import bootstrap
from common.log_utils import preview
from rag.factory import MotherFactory
from rag.pipeline.chunker import chunk
from rag.pipeline.indexer import index
from rag.pipeline.parser import parse
from rag.pipeline.rerank import rerank_results
from rag.pipeline.report import write_results_md
from rag.pipeline.store import init_db
from rag.pipeline.synthesize import synthesize_answer
from rag.search.runner import embed_query, run_search

bootstrap("INFO")
logger = logging.getLogger(__name__)


def main() -> None:
    # ────────────────────────── Choosing the Mother Instance ─────────────────────
    # The MotherInstance returned by `interactive_pick()` bundles four locked
    # picks: an embedding instance (template + tokenizer + provider + metric),
    # a search instance (the subset of searchers to run), and optionally a
    # reranker instance and an LLM instance. Everything downstream flows from
    # it. To skip the interactive picker in code, use
    # `MotherFactory.from_saved(name)` instead.

    mother = MotherFactory.interactive_pick()
    logger.info("alpha_test mother=%s", mother.describe())

    embedding = mother.embedding

    # ────────────────────────── Parsing the Document ────────────────────────────

    file_path = "data/bcs.pdf"

    fingerprint = parse(file_path)
    logger.info("alpha_test parsed file=%s fingerprint=%s", file_path, fingerprint)

    # ────────────────────────── Chunking the Document ───────────────────────────

    init_db()
    chunks = chunk(fingerprint, embedding.count_tokens, embedding.model_key)
    logger.info("alpha_test built/loaded chunks=%d", len(chunks))

    # ────────────────────────── Indexing the Document Chunks ─────────────────────

    index(chunks, embedding)

    # ────────────────────────── Preparing the Query ──────────────────────────────

    #query = "Mike calls Jimmy for help, and Jimmy comes to the rescue"
    query = "Mike calls Jimmy for help, and Jimmy comes to the rescue! explain what happened?"
    start_time = time.perf_counter()
    query_vector = embed_query(embedding, query)
    end_time = time.perf_counter()
    logger.info("alpha_test embed_query time=%s", end_time - start_time)

    # ────────────────────────── Running the Searches ─────────────────────────────
    # Only the strategies selected in the mother's search instance run.

    k = max(1, min(20, len(chunks) // 4))

    strategies = mother.search.strategies
    if not strategies:
        raise ValueError("Mother instance has an empty search instance; nothing to run.")

    runs = []
    for name in strategies:
        search_time, results = run_search(
            name, query_vector, query, fingerprint, embedding, k
        )

        # ────────────────── Reranking the Top-K (per searcher) ────────────────────
        # No-op when `mother.reranker is None`; otherwise hydrates chunk text
        # from SQLite and replaces this searcher's results with a cross-encoder
        # reranked ordering. The search elapsed time stays as the bi-encoder
        # search time — rerank time is logged separately.
        _, results = rerank_results(
            name, results, query, fingerprint, embedding, mother.reranker, k
        )

        runs.append((name, search_time, results))

    # ────────────────────────── Optional LLM Answer Synthesis ────────────────────
    # When the mother instance carries an LLM, synthesise an answer from the
    # top results of the first searcher run so it can be embedded at the top of
    # the report. No-op (returns None) otherwise.

    first_name, _, first_results = runs[0]
    resp = synthesize_answer(first_results, query, fingerprint, embedding, mother.llm)
    llm_answer = None
    if resp is not None:
        llm_answer = resp.text
        logger.info(
            "alpha_test llm answer searcher=%s model=%s tokens=%s text='%s'",
            first_name,
            resp.model,
            resp.tokens_used,
            preview(resp.text),
        )

    # ────────────────────────── Writing the Markdown Report ──────────────────────

    reranker_label = mother.reranker.provider_id if mother.reranker is not None else None
    report_path = write_results_md(
        query,
        fingerprint,
        embedding.model_key,
        runs,
        k,
        reranker_label=reranker_label,
        llm_answer=llm_answer,
    )
    logger.info("alpha_test report written path=%s", report_path)


if __name__ == "__main__":
    main()
