"""
Interactive picker for selecting (or assembling) a Mother Instance.

A mother instance ties together one embedding instance, one search
instance (a subset of searchers to run), and optionally one reranker
instance and one LLM instance. This is the single object the pipeline
runs on.

Flow:

  1. Show saved mother instances (pick one to run).
  2. Offer "assemble a new mother instance", which walks each slot:
     pick an existing sub-instance or create one on the fly (which in
     turn picks a template + specific picks and saves it).
  3. Admin submenu to register templates.

Factories are imported lazily inside functions to avoid a circular
import (`MotherFactory.interactive_pick()` calls into this module).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from rag.factory.instance import MotherInstance
from rag.registry.loader import REGISTRY_PATH, load_registry
from rag.registry.schema import EmbeddingTemplate, Registry, Status
from rag.search.distance_metrics.base_distance_metric import MetricKind

logger = logging.getLogger(__name__)


def run_picker() -> MotherInstance:
    while True:
        registry = load_registry()
        action, payload = _print_top_menu(registry)

        if action == "use_mother":
            from rag.factory.factory import MotherFactory

            print(f"\nLoading mother instance '{payload}' ...")
            return MotherFactory.from_saved(payload)

        if action == "build_mother":
            mother = _build_mother_flow()
            if mother is None:
                continue
            return mother

        if action == "admin":
            _admin_menu()
            continue

        if action == "quit":
            raise SystemExit(0)


def _print_top_menu(registry: Registry) -> Tuple[str, Optional[str]]:
    print()
    print("=" * 72)
    print(" Multi RAG Engine — pick a mother instance")
    print(f" Registry: {REGISTRY_PATH}")
    print("=" * 72)

    options: List[Tuple[str, Optional[str]]] = []

    if registry.mother_instances:
        print("\nSaved mother instances:")
        for name, m in registry.mother_instances.items():
            options.append(("use_mother", name))
            extras = []
            if m.reranker_instance:
                extras.append(f"reranker={m.reranker_instance}")
            if m.llm_instance:
                extras.append(f"llm={m.llm_instance}")
            extra = ("  " + " ".join(extras)) if extras else ""
            print(
                f"  [{len(options)}] {name}    "
                f"embedding={m.embedding_instance}  search={m.search_instance}{extra}"
            )
    else:
        print("\n(no saved mother instances yet)")

    print("\nActions:")
    options.append(("build_mother", None))
    print(f"  [{len(options)}] Assemble a new mother instance")
    options.append(("admin", None))
    print(f"  [{len(options)}] Admin: register templates")
    options.append(("quit", None))
    print(f"  [{len(options)}] Quit")

    while True:
        raw = input(f"\nPick [1-{len(options)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'; try again.")


def _build_mother_flow() -> Optional[MotherInstance]:
    print("\n--- Assemble a new mother instance ---")

    embedding_name = _select_or_create_embedding()
    if embedding_name is None:
        print("  cancelled (embedding instance is required).")
        return None

    search_name = _select_or_create_search()
    if search_name is None:
        print("  cancelled (search instance is required).")
        return None

    reranker_name = _select_or_create_reranker()
    llm_name = _select_or_create_llm()

    from rag.factory.factory import (
        EmbeddingFactory,
        LLMFactory,
        MotherFactory,
        RerankerFactory,
    )
    from rag.registry.wizard import save_mother_instance_flow

    mother_name: Optional[str] = None
    answer = input("\nSave this as a named mother instance? [y/N]: ").strip().lower()
    if answer == "y":
        mother_name = save_mother_instance_flow(
            embedding_instance=embedding_name,
            search_instance=search_name,
            reranker_instance=reranker_name,
            llm_instance=llm_name,
        )

    print("\nBuilding mother instance ...")
    embedding = EmbeddingFactory.from_saved_instance(embedding_name)
    registry = load_registry()
    search = MotherFactory.build_search(registry, search_name)
    reranker = (
        RerankerFactory.from_saved_instance(reranker_name) if reranker_name else None
    )
    llm = LLMFactory.from_saved_instance(llm_name) if llm_name else None
    return MotherInstance(
        name=mother_name,
        embedding=embedding,
        search=search,
        reranker=reranker,
        llm=llm,
    )


# ─────────────────────────── per-slot selection ─────────────────────────────


def _select_or_create_embedding() -> Optional[str]:
    registry = load_registry()
    existing = list(registry.embedding_instances.keys())
    choice = _slot_menu("embedding instance", existing, optional=False)
    if choice == "cancel":
        return None
    if choice == "create":
        return _create_embedding_instance()
    return choice


def _select_or_create_reranker() -> Optional[str]:
    registry = load_registry()
    existing = list(registry.reranker_instances.keys())
    choice = _slot_menu("reranker instance", existing, optional=True)
    if choice in ("cancel", "skip"):
        return None
    if choice == "create":
        return _create_reranker_instance()
    return choice


def _select_or_create_llm() -> Optional[str]:
    registry = load_registry()
    existing = list(registry.llm_instances.keys())
    choice = _slot_menu("LLM instance", existing, optional=True)
    if choice in ("cancel", "skip"):
        return None
    if choice == "create":
        return _create_llm_instance()
    return choice


def _select_or_create_search() -> Optional[str]:
    registry = load_registry()
    existing = list(registry.search_instances.keys())
    choice = _slot_menu("search instance", existing, optional=False)
    if choice == "cancel":
        return None
    if choice == "create":
        return _create_search_instance()
    return choice


def _slot_menu(label: str, existing: List[str], optional: bool) -> str:
    """
    Return one of: an existing instance name, "create", "skip" (optional
    only), or "cancel".
    """
    print(f"\nPick a {label}:")
    for idx, name in enumerate(existing, start=1):
        print(f"  [{idx}] {name}")
    create_idx = len(existing) + 1
    print(f"  [{create_idx}] + create a new {label}")
    hint = f"1-{create_idx}"
    if optional:
        print(f"  [s] skip (no {label})")
        hint += ", s"
    print(f"  [c] cancel")
    hint += ", c"

    while True:
        raw = input(f"  pick [{hint}]: ").strip().lower()
        if raw == "c":
            return "cancel"
        if optional and raw == "s":
            return "skip"
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
            if idx == len(existing):
                return "create"
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'; try again.")


# ─────────────────────────── on-the-fly creation ────────────────────────────


def _create_embedding_instance() -> Optional[str]:
    from rag.registry.wizard import (
        register_embedding_template_flow,
        save_embedding_instance_flow,
    )

    registry = load_registry()
    template_key = _pick_template_key(
        "embedding",
        list(registry.embedding_templates.keys()),
        register_embedding_template_flow,
    )
    if template_key is None:
        return None
    registry = load_registry()
    picks = _pick_within_embedding_template(registry.embedding_templates[template_key])
    if picks is None:
        return None
    tokenizer_id, provider_id, metric_override = picks
    return save_embedding_instance_flow(
        template_key=template_key,
        tokenizer_id=tokenizer_id,
        provider_id=provider_id,
        metric_override=metric_override,
    )


def _create_reranker_instance() -> Optional[str]:
    from rag.registry.wizard import (
        register_reranker_template_flow,
        save_reranker_instance_flow,
    )

    registry = load_registry()
    template_key = _pick_template_key(
        "reranker",
        list(registry.reranker_templates.keys()),
        register_reranker_template_flow,
    )
    if template_key is None:
        return None
    registry = load_registry()
    template = registry.reranker_templates[template_key]
    provider_id = _pick_one(
        title=f"Providers in reranker template '{template_key}':",
        entries=[
            (
                pid,
                f"kind={s.kind}  model_id={s.model_id}  "
                f"strategy={s.score_strategy or '-'}  status={s.status.value}",
            )
            for pid, s in template.providers.items()
        ],
    )
    if provider_id is None:
        return None
    return save_reranker_instance_flow(template_key=template_key, provider_id=provider_id)


def _create_llm_instance() -> Optional[str]:
    from rag.registry.wizard import (
        register_llm_template_flow,
        save_llm_instance_flow,
    )

    registry = load_registry()
    template_key = _pick_template_key(
        "llm",
        list(registry.llm_templates.keys()),
        register_llm_template_flow,
    )
    if template_key is None:
        return None
    registry = load_registry()
    template = registry.llm_templates[template_key]
    provider_id = _pick_one(
        title=f"Providers in LLM template '{template_key}':",
        entries=[
            (
                pid,
                f"kind={s.kind}  model_id={s.model_id}  "
                f"base_url={s.default_base_url}  status={s.status.value}",
            )
            for pid, s in template.providers.items()
        ],
    )
    if provider_id is None:
        return None
    return save_llm_instance_flow(template_key=template_key, provider_id=provider_id)


def _create_search_instance() -> Optional[str]:
    from rag.registry.wizard import save_search_instance_flow

    registry = load_registry()
    strategies = _multi_select_strategies(registry.search_strategies)
    if not strategies:
        return None
    return save_search_instance_flow(strategies)


def _pick_template_key(kind: str, keys: List[str], register_flow) -> Optional[str]:
    if not keys:
        print(f"\nNo {kind} templates registered yet.")
        answer = input(f"  register a {kind} template now? [y/N]: ").strip().lower()
        if answer != "y":
            return None
        return register_flow()

    print(f"\nAvailable {kind} templates:")
    for idx, key in enumerate(keys, start=1):
        print(f"  [{idx}] {key}")
    reg_idx = len(keys) + 1
    print(f"  [{reg_idx}] + register a new {kind} template")
    print(f"  [c] cancel")
    while True:
        raw = input(f"  pick [1-{reg_idx}, c]: ").strip().lower()
        if raw == "c":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            if idx == len(keys):
                return register_flow()
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'; try again.")


def _pick_within_embedding_template(
    template: EmbeddingTemplate,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return (tokenizer_id, provider_id, metric_override) or None to cancel."""
    if not template.tokenizers or not template.providers:
        print(
            "\nTemplate is incomplete (no tokenizers or no providers). "
            "Add some via the register flow first."
        )
        return None

    print(f"\nTemplate: {template.model_key}    metric={template.metric}    dim={template.dimension}")

    tokenizer_id = _pick_one(
        title="Tokenizers in this template:",
        entries=[
            (
                tid,
                f"kind={s.kind}  repo={s.repo}  status={s.status.value}",
            )
            for tid, s in template.tokenizers.items()
        ],
    )
    if tokenizer_id is None:
        return None

    provider_id = _pick_one(
        title="Providers in this template:",
        entries=[
            (
                pid,
                f"kind={s.kind}  model_id={s.model_id}  "
                f"base_url={s.default_base_url}  status={s.status.value}",
            )
            for pid, s in template.providers.items()
        ],
    )
    if provider_id is None:
        return None

    metric_override: Optional[str] = None
    print(f"\nDistance metric default: {template.metric}")
    answer = input("Override the default metric for this run? [y/N]: ").strip().lower()
    if answer == "y":
        valid = [m.value for m in MetricKind]
        while True:
            raw = input(f"  enter one of {valid}: ").strip().lower()
            if raw in valid:
                metric_override = raw
                break
            print(f"  invalid metric '{raw}'.")

    return tokenizer_id, provider_id, metric_override


def _multi_select_strategies(strategies: List[str]) -> List[str]:
    """Prompt for a subset of search strategies; returns the chosen list (may be empty on cancel)."""
    print("\nAvailable search strategies:")
    for idx, name in enumerate(strategies, start=1):
        print(f"  [{idx}] {name}")
    print("  [a] all")
    print("  [c] cancel")
    while True:
        raw = input(
            f"  pick a subset (comma-separated 1-{len(strategies)}, or a / c): "
        ).strip().lower()
        if raw == "c":
            return []
        if raw == "a":
            return list(strategies)
        chosen: List[str] = []
        ok = True
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part) - 1
            except ValueError:
                ok = False
                break
            if not (0 <= idx < len(strategies)):
                ok = False
                break
            if strategies[idx] not in chosen:
                chosen.append(strategies[idx])
        if ok and chosen:
            return chosen
        print("  invalid selection; try again.")


def _pick_one(
    title: str,
    entries: List[Tuple[str, str]],
    allow_skip: bool = False,
) -> Optional[str]:
    print(f"\n{title}")
    for idx, (eid, descr) in enumerate(entries, start=1):
        marker = ""
        for status_val in (s.value for s in Status):
            if f"status={status_val}" in descr:
                if status_val == Status.STABLE.value:
                    marker = " (OK)"
                elif status_val in (
                    Status.UNREACHABLE.value,
                    Status.UNAVAILABLE.value,
                ):
                    marker = " (FAIL)"
                break
        print(f"  [{idx}] {eid}{marker}    {descr}")
    if allow_skip:
        print(f"  [s] skip (no selection)")
    print(f"  [c] cancel")

    hint = f"1-{len(entries)} or {'s, ' if allow_skip else ''}c"
    while True:
        raw = input(f"\n  pick [{hint}]: ").strip().lower()
        if raw == "c":
            return None
        if allow_skip and raw == "s":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(entries):
                return entries[idx][0]
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'; try again.")


def _admin_menu() -> None:
    from rag.registry.wizard import (
        register_embedding_template_flow,
        register_llm_template_flow,
        register_reranker_template_flow,
    )

    while True:
        print("\n--- Admin: register templates ---")
        print("  [1] Register an embedding template")
        print("  [2] Register a reranker template")
        print("  [3] Register an LLM template")
        print("  [b] Back")
        raw = input("  pick [1-3, b]: ").strip().lower()
        if raw == "b":
            return
        if raw == "1":
            register_embedding_template_flow()
        elif raw == "2":
            register_reranker_template_flow()
        elif raw == "3":
            register_llm_template_flow()
        else:
            print(f"  invalid choice '{raw}'.")
