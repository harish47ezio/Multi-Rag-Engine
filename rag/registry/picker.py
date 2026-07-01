"""
Interactive picker for selecting (or registering) an Instance.

Flow:

  1. Show saved instances (if any).
  2. Show templates available for ad-hoc build.
  3. Show the "register new template" option.
  4. Once a template is chosen, prompt for tokenizer + provider + optional
     metric override, optionally save as a named instance, and return the
     resulting Instance.

The picker imports the factory lazily inside functions to avoid a
circular import (the factory exposes `interactive_pick()` which in turn
imports this module).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from rag.factory.instance import Instance
from rag.registry.loader import REGISTRY_PATH, load_registry
from rag.registry.schema import Registry, Status, Template
from rag.search.distance_metrics.base_distance_metric import MetricKind

logger = logging.getLogger(__name__)


def run_picker() -> Instance:
    from rag.factory.factory import EmbedderFactory
    from rag.registry.wizard import register_template_flow, save_instance_flow

    while True:
        registry = load_registry()
        action, payload = _print_top_menu(registry)

        if action == "use_instance":
            name = payload
            print(f"\nLoading saved instance '{name}' ...")
            return EmbedderFactory.from_saved_instance(name)

        if action == "build_from_template":
            template = registry.templates[payload]
            picks = _pick_within_template(template)
            if picks is None:
                continue
            tokenizer_id, provider_id, metric_override, reranker_id = picks
            instance = EmbedderFactory.from_template(
                model_key=template.model_key,
                tokenizer_id=tokenizer_id,
                provider_id=provider_id,
                metric_override=metric_override,
                reranker_id=reranker_id,
            )
            answer = input("\nSave this as a named instance? [y/N]: ").strip().lower()
            if answer == "y":
                save_instance_flow(
                    template_key=template.model_key,
                    tokenizer_id=tokenizer_id,
                    provider_id=provider_id,
                    metric_override=metric_override,
                    reranker_id=reranker_id,
                )
            return instance

        if action == "register":
            register_template_flow()
            continue

        if action == "quit":
            raise SystemExit(0)


def _print_top_menu(registry: Registry) -> Tuple[str, Optional[str]]:
    """Return ('use_instance', name) | ('build_from_template', key) | ('register', None) | ('quit', None)."""
    print()
    print("=" * 72)
    print(" Multi RAG Engine — pick an embedding instance")
    print(f" Registry: {REGISTRY_PATH}")
    print("=" * 72)

    options: list[Tuple[str, Optional[str], str]] = []

    if registry.instances:
        print("\nSaved instances:")
        for name, inst in registry.instances.items():
            override = (
                f" / metric={inst.metric_override}"
                if inst.metric_override
                else ""
            )
            options.append(("use_instance", name, ""))
            label = (
                f"  [{len(options)}] {name}    "
                f"{inst.template_key} / {inst.tokenizer_id} / {inst.provider_id}{override}"
            )
            print(label)

    if registry.templates:
        print("\nBuild a new instance from a template:")
        for key, t in registry.templates.items():
            options.append(("build_from_template", key, ""))
            tok_count = len(t.tokenizers)
            prov_count = len(t.providers)
            print(
                f"  [{len(options)}] {key}    "
                f"metric={t.metric}  dim={t.dimension}  "
                f"tokenizers={tok_count}  providers={prov_count}"
            )

    print("\nAdmin:")
    options.append(("register", None, ""))
    print(f"  [{len(options)}] + Register a new template")
    options.append(("quit", None, ""))
    print(f"  [{len(options)}] Quit")

    while True:
        raw = input("\nPick [1-{}]: ".format(len(options))).strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                action, payload, _ = options[idx]
                return action, payload
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'; try again.")


def _pick_within_template(
    template: Template,
) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
    """Return (tokenizer_id, provider_id, metric_override, reranker_id) or None to cancel."""
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
                f"kind={s.kind}  repo={s.repo}  status={s.status.value}"
                + (f"  last_checked={s.last_checked_at}" if s.last_checked_at else ""),
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
                f"base_url={s.default_base_url}  status={s.status.value}"
                + (f"  last_checked={s.last_checked_at}" if s.last_checked_at else ""),
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

    reranker_id: Optional[str] = None
    if template.rerankers:
        reranker_id = _pick_one(
            title="Rerankers in this template (or skip):",
            entries=[
                (
                    rid,
                    f"kind={s.kind}  model_id={s.model_id}  "
                    f"strategy={s.score_strategy or '-'}  status={s.status.value}"
                    + (f"  last_checked={s.last_checked_at}" if s.last_checked_at else ""),
                )
                for rid, s in template.rerankers.items()
            ],
            allow_skip=True,
        )

    return tokenizer_id, provider_id, metric_override, reranker_id


def _pick_one(
    title: str,
    entries: list[Tuple[str, str]],
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
                else:
                    marker = ""
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
