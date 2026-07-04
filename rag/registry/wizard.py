"""
Interactive flows for mutating the registry.

Register flows build and persist reusable *templates*:
  * `register_embedding_template_flow()` — model_key, metric, dimension,
    one+ tokenizers, one+ providers.
  * `register_reranker_template_flow()` — model_key, one+ reranker providers.
  * `register_llm_template_flow()` — model_key, one+ LLM providers.

Every entry is validated immediately after creation; failures are saved
too (status=unreachable / unavailable) so the user can fix them later.

Save flows persist named *instances* (locked picks). Validation has
already happened at instance construction time, so these are pure saves:
  * `save_embedding_instance_flow(...)`
  * `save_reranker_instance_flow(...)`
  * `save_llm_instance_flow(...)`
  * `save_search_instance_flow(...)`
  * `save_mother_instance_flow(...)`
"""

from __future__ import annotations

import logging
from typing import List, Optional

from rag.registry.loader import load_registry, save_registry
from rag.registry.schema import (
    EmbeddingInstanceSpec,
    EmbeddingTemplate,
    LLMInstanceSpec,
    LLMTemplate,
    MotherInstanceSpec,
    ProviderSpec,
    RerankerInstanceSpec,
    RerankerSpec,
    RerankerTemplate,
    SearchInstanceSpec,
    TokenizerSpec,
)
from rag.registry.validator import (
    now_iso,
    update_status_from_check,
    validate_provider,
    validate_reranker,
    validate_tokenizer,
)
from rag.search.distance_metrics.base_distance_metric import MetricKind

logger = logging.getLogger(__name__)


# ───────────────────────────── template register flows ──────────────────────


def register_embedding_template_flow() -> Optional[str]:
    """Interactive: build and persist a new embedding template. Returns its key or None."""
    print("\n--- Register a new embedding template ---")

    model_key = input("Model key (canonical name, e.g. 'qwen3-embedding-8b'): ").strip()
    if not model_key:
        print("  cancelled (empty model_key).")
        return None

    registry = load_registry()
    if model_key in registry.embedding_templates:
        answer = input(
            f"Embedding template '{model_key}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            return None

    valid_metrics = [m.value for m in MetricKind]
    while True:
        metric = input(f"Distance metric {valid_metrics}: ").strip().lower()
        if metric in valid_metrics:
            break
        print(f"  invalid metric '{metric}'.")

    dim_raw = input("Dimension (blank to probe at first embed): ").strip()
    dimension: Optional[int]
    if dim_raw == "":
        dimension = None
    else:
        try:
            dimension = int(dim_raw)
        except ValueError:
            print(f"  invalid integer '{dim_raw}'; treating as blank.")
            dimension = None

    template = EmbeddingTemplate(
        model_key=model_key,
        metric=metric,
        dimension=dimension,
    )

    print("\nAdd at least one tokenizer.")
    while True:
        added = _prompt_tokenizer(template)
        if added is None:
            if not template.tokenizers:
                print("  at least one tokenizer is required.")
                continue
            break
        more = input("  add another tokenizer? [y/N]: ").strip().lower()
        if more != "y":
            break

    print("\nAdd at least one provider.")
    while True:
        added = _prompt_embedding_provider(template)
        if added is None:
            if not template.providers:
                print("  at least one provider is required.")
                continue
            break
        more = input("  add another provider? [y/N]: ").strip().lower()
        if more != "y":
            break

    registry.embedding_templates[model_key] = template
    save_registry(registry)
    logger.info("register_embedding_template_flow saved model_key=%s", model_key)
    print(f"\nEmbedding template '{model_key}' saved.")
    return model_key


def register_reranker_template_flow() -> Optional[str]:
    """Interactive: build and persist a new reranker template. Returns its key or None."""
    print("\n--- Register a new reranker template ---")

    model_key = input("Reranker model key (e.g. 'bge-reranker-v2-m3'): ").strip()
    if not model_key:
        print("  cancelled (empty model_key).")
        return None

    registry = load_registry()
    if model_key in registry.reranker_templates:
        answer = input(
            f"Reranker template '{model_key}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            return None

    template = RerankerTemplate(model_key=model_key)

    print("\nAdd at least one reranker provider.")
    while True:
        added = _prompt_reranker_provider(template)
        if added is None:
            if not template.providers:
                print("  at least one provider is required.")
                continue
            break
        more = input("  add another provider? [y/N]: ").strip().lower()
        if more != "y":
            break

    registry.reranker_templates[model_key] = template
    save_registry(registry)
    logger.info("register_reranker_template_flow saved model_key=%s", model_key)
    print(f"\nReranker template '{model_key}' saved.")
    return model_key


def register_llm_template_flow() -> Optional[str]:
    """Interactive: build and persist a new LLM template. Returns its key or None."""
    print("\n--- Register a new LLM template ---")

    model_key = input("LLM model key (e.g. 'qwen3-next-80b'): ").strip()
    if not model_key:
        print("  cancelled (empty model_key).")
        return None

    registry = load_registry()
    if model_key in registry.llm_templates:
        answer = input(
            f"LLM template '{model_key}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            return None

    template = LLMTemplate(model_key=model_key)

    print("\nAdd at least one LLM provider.")
    while True:
        added = _prompt_llm_provider(template)
        if added is None:
            if not template.providers:
                print("  at least one provider is required.")
                continue
            break
        more = input("  add another provider? [y/N]: ").strip().lower()
        if more != "y":
            break

    registry.llm_templates[model_key] = template
    save_registry(registry)
    logger.info("register_llm_template_flow saved model_key=%s", model_key)
    print(f"\nLLM template '{model_key}' saved.")
    return model_key


# ─────────────────────────────── entry prompts ──────────────────────────────


def _prompt_tokenizer(template: EmbeddingTemplate) -> Optional[str]:
    tid = input("  Tokenizer id (short label, e.g. 'hf-qwen3-8b'): ").strip()
    if not tid:
        return None
    if tid in template.tokenizers:
        print(f"  tokenizer id '{tid}' already taken in this template.")
        return None

    kind = input("  Tokenizer kind [hf]: ").strip().lower() or "hf"
    repo = ""
    if kind == "hf":
        repo = input("  HuggingFace repo (e.g. 'Qwen/Qwen3-Embedding-8B'): ").strip()
        if not repo:
            print("  cancelled (empty repo).")
            return None
    else:
        print(f"  kind '{kind}' not supported yet; saving anyway with status=unavailable.")

    spec = TokenizerSpec(id=tid, kind=kind, repo=repo)
    print(f"  validating tokenizer '{tid}' ...")
    check = validate_tokenizer(spec)
    update_status_from_check(spec, check)
    print(f"    [{'OK' if check.ok else 'FAIL'}] {check.detail}")

    template.tokenizers[tid] = spec
    return tid


def _prompt_embedding_provider(template: EmbeddingTemplate) -> Optional[str]:
    pid = input("  Provider id (short label, e.g. 'ollama-local'): ").strip()
    if not pid:
        return None
    if pid in template.providers:
        print(f"  provider id '{pid}' already taken in this template.")
        return None

    kind = input("  Provider kind [ollama, hf]: ").strip().lower() or "ollama"
    model_id = input("  Provider-specific model id (e.g. 'qwen3-embedding:8b'): ").strip()
    if not model_id:
        print("  cancelled (empty model_id).")
        return None

    default_base_url, requires_api_key = _prompt_transport(kind)

    spec = ProviderSpec(
        id=pid,
        kind=kind,
        model_id=model_id,
        default_base_url=default_base_url,
        requires_api_key=requires_api_key,
    )
    print(f"  validating provider '{pid}' ...")
    check = validate_provider(spec)
    update_status_from_check(spec, check)
    print(f"    [{'OK' if check.ok else 'FAIL'}] {check.detail}")

    template.providers[pid] = spec
    return pid


def _prompt_reranker_provider(template: RerankerTemplate) -> Optional[str]:
    pid = input("  Provider id (short label, e.g. 'ollama-bge-rerank'): ").strip()
    if not pid:
        return None
    if pid in template.providers:
        print(f"  provider id '{pid}' already taken in this template.")
        return None

    kind = input("  Reranker kind [ollama, hf]: ").strip().lower() or "ollama"
    model_id = input(
        "  Model id (ollama tag or HF repo, e.g. 'linux6200/bge-reranker-v2-m3' / "
        "'BAAI/bge-reranker-v2-m3'): "
    ).strip()
    if not model_id:
        print("  cancelled (empty model_id).")
        return None

    default_base_url: Optional[str] = None
    requires_api_key = False
    score_strategy: Optional[str] = None
    if kind == "ollama":
        default_base_url, requires_api_key = _prompt_transport(kind)
        strategy_raw = (
            input("  Score strategy [embed, generate] (default 'embed'): ").strip().lower()
        )
        score_strategy = strategy_raw if strategy_raw in ("embed", "generate") else "embed"
    elif kind == "hf":
        score_strategy = None
    else:
        print(f"  kind '{kind}' not supported yet; saving anyway with status=unavailable.")

    spec = RerankerSpec(
        id=pid,
        kind=kind,
        model_id=model_id,
        default_base_url=default_base_url,
        requires_api_key=requires_api_key,
        score_strategy=score_strategy,
    )
    print(f"  validating reranker '{pid}' ...")
    check = validate_reranker(spec)
    update_status_from_check(spec, check)
    print(f"    [{'OK' if check.ok else 'FAIL'}] {check.detail}")

    template.providers[pid] = spec
    return pid


def _prompt_llm_provider(template: LLMTemplate) -> Optional[str]:
    pid = input("  Provider id (short label, e.g. 'ollama-cloud'): ").strip()
    if not pid:
        return None
    if pid in template.providers:
        print(f"  provider id '{pid}' already taken in this template.")
        return None

    kind = input("  Provider kind [ollama]: ").strip().lower() or "ollama"
    model_id = input("  Provider-specific model id (e.g. 'qwen3-coder:480b'): ").strip()
    if not model_id:
        print("  cancelled (empty model_id).")
        return None

    default_base_url, requires_api_key = _prompt_transport(kind)

    spec = ProviderSpec(
        id=pid,
        kind=kind,
        model_id=model_id,
        default_base_url=default_base_url,
        requires_api_key=requires_api_key,
    )
    print(f"  validating LLM provider '{pid}' ...")
    check = validate_provider(spec)
    update_status_from_check(spec, check)
    print(f"    [{'OK' if check.ok else 'FAIL'}] {check.detail}")

    template.providers[pid] = spec
    return pid


def _prompt_transport(kind: str):
    """Shared base-url + api-key prompt for network-served kinds."""
    if kind == "ollama":
        base_url_raw = input("  Base URL [http://localhost:11434]: ").strip()
        default_base_url = base_url_raw or "http://localhost:11434"
        requires_api_key = input("  Requires API key? [y/N]: ").strip().lower() == "y"
        return default_base_url, requires_api_key
    if kind == "hf":
        base_url_raw = input("  Base URL (blank for HF default): ").strip()
        default_base_url = base_url_raw or None
        requires_api_key = input("  Requires API key? [y/N]: ").strip().lower() == "y"
        return default_base_url, requires_api_key
    print(f"  kind '{kind}' not supported yet; saving anyway with status=unavailable.")
    return None, False


# ─────────────────────────────── save flows ─────────────────────────────────


def save_embedding_instance_flow(
    template_key: str,
    tokenizer_id: str,
    provider_id: str,
    metric_override: Optional[str],
) -> Optional[str]:
    """Persist a named embedding instance. Returns the name, or None on cancel."""
    while True:
        name = input("Embedding instance name (e.g. 'fast-qwen'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.embedding_instances:
            answer = input(
                f"  instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.embedding_instances[name] = EmbeddingInstanceSpec(
            name=name,
            template_key=template_key,
            tokenizer_id=tokenizer_id,
            provider_id=provider_id,
            metric_override=metric_override,
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info("save_embedding_instance_flow saved name=%s", name)
        print(f"  saved embedding instance '{name}'.")
        return name


def save_reranker_instance_flow(
    template_key: str,
    provider_id: str,
) -> Optional[str]:
    """Persist a named reranker instance. Returns the name, or None on cancel."""
    while True:
        name = input("Reranker instance name (e.g. 'bge-hf'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.reranker_instances:
            answer = input(
                f"  instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.reranker_instances[name] = RerankerInstanceSpec(
            name=name,
            template_key=template_key,
            provider_id=provider_id,
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info("save_reranker_instance_flow saved name=%s", name)
        print(f"  saved reranker instance '{name}'.")
        return name


def save_llm_instance_flow(
    template_key: str,
    provider_id: str,
) -> Optional[str]:
    """Persist a named LLM instance. Returns the name, or None on cancel."""
    while True:
        name = input("LLM instance name (e.g. 'qwen-cloud'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.llm_instances:
            answer = input(
                f"  instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.llm_instances[name] = LLMInstanceSpec(
            name=name,
            template_key=template_key,
            provider_id=provider_id,
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info("save_llm_instance_flow saved name=%s", name)
        print(f"  saved LLM instance '{name}'.")
        return name


def save_search_instance_flow(strategies: List[str]) -> Optional[str]:
    """Persist a named search instance (subset of strategies). Returns name or None."""
    while True:
        name = input("Search instance name (e.g. 'ann-vs-knn'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.search_instances:
            answer = input(
                f"  instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.search_instances[name] = SearchInstanceSpec(
            name=name,
            strategies=list(strategies),
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info("save_search_instance_flow saved name=%s strategies=%s", name, strategies)
        print(f"  saved search instance '{name}'.")
        return name


def save_mother_instance_flow(
    embedding_instance: str,
    search_instance: str,
    reranker_instance: Optional[str],
    llm_instance: Optional[str],
) -> Optional[str]:
    """Persist a named mother instance. Returns the name, or None on cancel."""
    while True:
        name = input("Mother instance name (e.g. 'default'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.mother_instances:
            answer = input(
                f"  mother instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.mother_instances[name] = MotherInstanceSpec(
            name=name,
            embedding_instance=embedding_instance,
            search_instance=search_instance,
            reranker_instance=reranker_instance,
            llm_instance=llm_instance,
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info("save_mother_instance_flow saved name=%s", name)
        print(f"  saved mother instance '{name}'.")
        return name
