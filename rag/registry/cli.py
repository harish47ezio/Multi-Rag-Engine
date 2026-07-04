"""
CLI subcommands for the registry.

Usage:
    python -m rag.registry list
    python -m rag.registry register-template     # embedding template
    python -m rag.registry register-reranker
    python -m rag.registry register-llm
    python -m rag.registry save-mother
    python -m rag.registry validate <template_key>
    python -m rag.registry refresh
"""

from __future__ import annotations

import argparse
import sys

from common.log_utils import setup_logging
from rag.registry.loader import REGISTRY_PATH, load_registry, save_registry
from rag.registry.validator import (
    validate_full_embedding_template,
    validate_full_llm_template,
    validate_full_reranker_template,
)


def cmd_list(_args: argparse.Namespace) -> int:
    registry = load_registry()
    print(f"Registry: {REGISTRY_PATH}")
    print(f"  schema_version: {registry.schema_version}")

    print(f"\n  embedding_templates: {len(registry.embedding_templates)}")
    for tk, t in registry.embedding_templates.items():
        print(
            f"    - {tk}   metric={t.metric}  dim={t.dimension}  "
            f"tokenizers={len(t.tokenizers)}  providers={len(t.providers)}"
        )
        for tid, s in t.tokenizers.items():
            print(f"        tokenizer {tid}: {s.status.value}    repo={s.repo}")
        for pid, s in t.providers.items():
            print(
                f"        provider  {pid}: {s.status.value}    "
                f"model_id={s.model_id}  base_url={s.default_base_url}"
            )

    print(f"\n  reranker_templates: {len(registry.reranker_templates)}")
    for tk, t in registry.reranker_templates.items():
        print(f"    - {tk}   providers={len(t.providers)}")
        for pid, s in t.providers.items():
            print(
                f"        provider  {pid}: {s.status.value}    kind={s.kind}  "
                f"model_id={s.model_id}  strategy={s.score_strategy or '-'}"
            )

    print(f"\n  llm_templates: {len(registry.llm_templates)}")
    for tk, t in registry.llm_templates.items():
        print(f"    - {tk}   providers={len(t.providers)}")
        for pid, s in t.providers.items():
            print(
                f"        provider  {pid}: {s.status.value}    "
                f"model_id={s.model_id}  base_url={s.default_base_url}"
            )

    print(f"\n  search_strategies: {registry.search_strategies}")

    print(f"\n  embedding_instances: {len(registry.embedding_instances)}")
    for name, i in registry.embedding_instances.items():
        override = f"  metric_override={i.metric_override}" if i.metric_override else ""
        print(
            f"    - {name}   template={i.template_key}  "
            f"tokenizer={i.tokenizer_id}  provider={i.provider_id}{override}"
        )

    print(f"  reranker_instances: {len(registry.reranker_instances)}")
    for name, i in registry.reranker_instances.items():
        print(f"    - {name}   template={i.template_key}  provider={i.provider_id}")

    print(f"  llm_instances: {len(registry.llm_instances)}")
    for name, i in registry.llm_instances.items():
        print(f"    - {name}   template={i.template_key}  provider={i.provider_id}")

    print(f"  search_instances: {len(registry.search_instances)}")
    for name, i in registry.search_instances.items():
        print(f"    - {name}   strategies={i.strategies}")

    print(f"\n  mother_instances: {len(registry.mother_instances)}")
    for name, m in registry.mother_instances.items():
        print(
            f"    - {name}   embedding={m.embedding_instance}  search={m.search_instance}"
            f"  reranker={m.reranker_instance or '-'}  llm={m.llm_instance or '-'}"
        )
    return 0


def cmd_register_template(_args: argparse.Namespace) -> int:
    from rag.registry.wizard import register_embedding_template_flow

    register_embedding_template_flow()
    return 0


def cmd_register_reranker(_args: argparse.Namespace) -> int:
    from rag.registry.wizard import register_reranker_template_flow

    register_reranker_template_flow()
    return 0


def cmd_register_llm(_args: argparse.Namespace) -> int:
    from rag.registry.wizard import register_llm_template_flow

    register_llm_template_flow()
    return 0


def cmd_save_mother(_args: argparse.Namespace) -> int:
    """Interactive: assemble and save a mother instance (delegates to the picker flow)."""
    from rag.registry.picker import _build_mother_flow

    mother = _build_mother_flow()
    if mother is None:
        print("Cancelled.")
        return 0
    print(f"\nAssembled: {mother.describe()}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry = load_registry()
    if args.template_key in registry.embedding_templates:
        report = validate_full_embedding_template(
            registry.embedding_templates[args.template_key]
        )
    elif args.template_key in registry.reranker_templates:
        report = validate_full_reranker_template(
            registry.reranker_templates[args.template_key]
        )
    elif args.template_key in registry.llm_templates:
        report = validate_full_llm_template(registry.llm_templates[args.template_key])
    else:
        print(f"Template '{args.template_key}' not found in any catalogue.")
        return 1
    save_registry(registry)
    for c in report.checks:
        marker = "OK" if c.ok else "FAIL"
        print(f"  [{marker}] {c.name}: {c.detail}")
    return 0 if report.ok else 2


def cmd_refresh(_args: argparse.Namespace) -> int:
    registry = load_registry()
    any_fail = False
    any_template = False

    for tk, t in registry.embedding_templates.items():
        any_template = True
        print(f"\nRefreshing embedding template '{tk}' ...")
        report = validate_full_embedding_template(t)
        _print_report(report)
        any_fail = any_fail or not report.ok

    for tk, t in registry.reranker_templates.items():
        any_template = True
        print(f"\nRefreshing reranker template '{tk}' ...")
        report = validate_full_reranker_template(t)
        _print_report(report)
        any_fail = any_fail or not report.ok

    for tk, t in registry.llm_templates.items():
        any_template = True
        print(f"\nRefreshing LLM template '{tk}' ...")
        report = validate_full_llm_template(t)
        _print_report(report)
        any_fail = any_fail or not report.ok

    if not any_template:
        print("No templates to refresh.")
        return 0

    save_registry(registry)
    return 0 if not any_fail else 2


def _print_report(report) -> None:
    for c in report.checks:
        marker = "OK" if c.ok else "FAIL"
        print(f"  [{marker}] {c.name}: {c.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.registry",
        description="Multi RAG Engine — registry CLI",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("list", help="Show all templates and saved instances").set_defaults(
        func=cmd_list
    )
    subs.add_parser(
        "register-template", help="Interactively register a new embedding template"
    ).set_defaults(func=cmd_register_template)
    subs.add_parser(
        "register-reranker", help="Interactively register a new reranker template"
    ).set_defaults(func=cmd_register_reranker)
    subs.add_parser(
        "register-llm", help="Interactively register a new LLM template"
    ).set_defaults(func=cmd_register_llm)
    subs.add_parser(
        "save-mother", help="Interactively assemble and save a mother instance"
    ).set_defaults(func=cmd_save_mother)

    p_validate = subs.add_parser(
        "validate", help="Re-validate every entry in one template (any kind)"
    )
    p_validate.add_argument("template_key", type=str)
    p_validate.set_defaults(func=cmd_validate)

    subs.add_parser(
        "refresh", help="Re-validate every template in the registry"
    ).set_defaults(func=cmd_refresh)
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging("INFO")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
