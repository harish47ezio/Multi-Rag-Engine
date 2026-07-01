"""
Interactive flows for mutating the registry:

  * `register_template_flow()` — walk the user through declaring a new
    template (model_key, metric, dimension, one or more tokenizers, one
    or more providers). Each entry is validated *immediately* after
    creation; failures are saved as well (status=unreachable / unavailable)
    so the user can come back later when their environment is fixed.

  * `save_instance_flow(template_key, tokenizer_id, provider_id, metric_override?)`
    — persist a named pick. Validation has already happened at instance
    construction time; this is a pure save.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag.registry.loader import load_registry, save_registry
from rag.registry.schema import (
    ProviderSpec,
    RerankerSpec,
    SavedInstance,
    Status,
    Template,
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


def register_template_flow() -> Optional[str]:
    """Interactive: build and persist a new Template. Returns its key, or None on cancel."""
    print("\n--- Register a new template ---")

    model_key = input("Model key (canonical name, e.g. 'qwen3-embedding-8b'): ").strip()
    if not model_key:
        print("  cancelled (empty model_key).")
        return None

    registry = load_registry()
    if model_key in registry.templates:
        answer = input(
            f"Template '{model_key}' already exists. Overwrite? [y/N]: "
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

    template = Template(
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
        added = _prompt_provider(template)
        if added is None:
            if not template.providers:
                print("  at least one provider is required.")
                continue
            break
        more = input("  add another provider? [y/N]: ").strip().lower()
        if more != "y":
            break

    print("\nAdd one or more rerankers (optional, press Enter to skip).")
    while True:
        added = _prompt_reranker(template)
        if added is None:
            break
        more = input("  add another reranker? [y/N]: ").strip().lower()
        if more != "y":
            break

    registry.templates[model_key] = template
    save_registry(registry)
    logger.info("register_template_flow saved model_key=%s", model_key)
    print(f"\nTemplate '{model_key}' saved.")
    _print_template_status(template)
    return model_key


def _prompt_tokenizer(template: Template) -> Optional[str]:
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
    marker = "OK" if check.ok else "FAIL"
    print(f"    [{marker}] {check.detail}")

    template.tokenizers[tid] = spec
    return tid


def _prompt_provider(template: Template) -> Optional[str]:
    pid = input("  Provider id (short label, e.g. 'ollama-local'): ").strip()
    if not pid:
        return None
    if pid in template.providers:
        print(f"  provider id '{pid}' already taken in this template.")
        return None

    kind = input("  Provider kind [ollama]: ").strip().lower() or "ollama"
    model_id = input("  Provider-specific model id (e.g. 'qwen3-embedding:8b'): ").strip()
    if not model_id:
        print("  cancelled (empty model_id).")
        return None

    default_base_url: Optional[str] = None
    requires_api_key = False
    if kind == "ollama":
        base_url_raw = input(
            "  Base URL [http://localhost:11434]: "
        ).strip()
        default_base_url = base_url_raw or "http://localhost:11434"
        requires_api_key = (
            input("  Requires API key? [y/N]: ").strip().lower() == "y"
        )
    else:
        print(f"  kind '{kind}' not supported yet; saving anyway with status=unavailable.")

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
    marker = "OK" if check.ok else "FAIL"
    print(f"    [{marker}] {check.detail}")

    template.providers[pid] = spec
    return pid


def _prompt_reranker(template: Template) -> Optional[str]:
    rid = input("  Reranker id (short label, e.g. 'ollama-bge-rerank'): ").strip()
    if not rid:
        return None
    if rid in template.rerankers:
        print(f"  reranker id '{rid}' already taken in this template.")
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
        base_url_raw = input(
            "  Base URL [http://localhost:11434]: "
        ).strip()
        default_base_url = base_url_raw or "http://localhost:11434"
        requires_api_key = (
            input("  Requires API key? [y/N]: ").strip().lower() == "y"
        )
        strategy_raw = (
            input("  Score strategy [embed, generate] (default 'embed'): ").strip().lower()
        )
        score_strategy = strategy_raw if strategy_raw in ("embed", "generate") else "embed"
    elif kind == "hf":
        score_strategy = None
    else:
        print(f"  kind '{kind}' not supported yet; saving anyway with status=unavailable.")

    spec = RerankerSpec(
        id=rid,
        kind=kind,
        model_id=model_id,
        default_base_url=default_base_url,
        requires_api_key=requires_api_key,
        score_strategy=score_strategy,
    )
    print(f"  validating reranker '{rid}' ...")
    check = validate_reranker(spec)
    update_status_from_check(spec, check)
    marker = "OK" if check.ok else "FAIL"
    print(f"    [{marker}] {check.detail}")

    template.rerankers[rid] = spec
    return rid


def _print_template_status(template: Template) -> None:
    print("\nTemplate summary:")
    print(f"  model_key = {template.model_key}")
    print(f"  metric    = {template.metric}")
    print(f"  dimension = {template.dimension}")
    for tid, s in template.tokenizers.items():
        print(f"  tokenizer {tid}: status={s.status.value}  repo={s.repo}")
    for pid, s in template.providers.items():
        print(
            f"  provider  {pid}: status={s.status.value}  "
            f"model_id={s.model_id}  base_url={s.default_base_url}"
        )
    for rid, s in template.rerankers.items():
        print(
            f"  reranker  {rid}: status={s.status.value}  kind={s.kind}  "
            f"model_id={s.model_id}  strategy={s.score_strategy or '-'}"
        )


def save_instance_flow(
    template_key: str,
    tokenizer_id: str,
    provider_id: str,
    metric_override: Optional[str],
    reranker_id: Optional[str] = None,
) -> Optional[str]:
    """Persist a SavedInstance under a user-chosen name. Returns the name, or None on cancel."""
    while True:
        name = input("Instance name (short, e.g. 'fast-qwen'): ").strip()
        if not name:
            print("  cancelled (empty name).")
            return None
        registry = load_registry()
        if name in registry.instances:
            answer = input(
                f"  instance '{name}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if answer != "y":
                continue
        registry.instances[name] = SavedInstance(
            name=name,
            template_key=template_key,
            tokenizer_id=tokenizer_id,
            provider_id=provider_id,
            metric_override=metric_override,
            reranker_id=reranker_id,
            created_at=now_iso(),
        )
        save_registry(registry)
        logger.info(
            "save_instance_flow saved name=%s template=%s tokenizer=%s provider=%s reranker=%s",
            name,
            template_key,
            tokenizer_id,
            provider_id,
            reranker_id,
        )
        print(f"  saved instance '{name}'.")
        return name
