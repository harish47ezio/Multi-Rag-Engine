"""
CLI subcommands for the registry.

Usage:
    python -m rag.registry list
    python -m rag.registry register-template
    python -m rag.registry save-instance
    python -m rag.registry validate <template_key>
    python -m rag.registry refresh
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.log_utils import setup_logging
from rag.registry.loader import REGISTRY_PATH, load_registry, save_registry
from rag.registry.validator import validate_full_template

logger = logging.getLogger(__name__)


def cmd_list(_args: argparse.Namespace) -> int:
    registry = load_registry()
    print(f"Registry: {REGISTRY_PATH}")
    print(f"  schema_version: {registry.schema_version}")
    print(f"  templates: {len(registry.templates)}")
    for tk, t in registry.templates.items():
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
    print(f"  instances: {len(registry.instances)}")
    for name, i in registry.instances.items():
        override = f"  metric_override={i.metric_override}" if i.metric_override else ""
        print(
            f"    - {name}   template={i.template_key}  "
            f"tokenizer={i.tokenizer_id}  provider={i.provider_id}{override}"
        )
    return 0


def cmd_register_template(_args: argparse.Namespace) -> int:
    from rag.registry.wizard import register_template_flow

    register_template_flow()
    return 0


def cmd_save_instance(_args: argparse.Namespace) -> int:
    """Interactive: pick template + tokenizer + provider, optional metric override, save as named instance."""
    from rag.registry.picker import _pick_within_template
    from rag.registry.wizard import save_instance_flow

    registry = load_registry()
    if not registry.templates:
        print("No templates in registry. Run `register-template` first.")
        return 1

    print("Available templates:")
    keys = list(registry.templates.keys())
    for idx, key in enumerate(keys, start=1):
        print(f"  [{idx}] {key}")
    while True:
        raw = input(f"Pick a template [1-{len(keys)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                template_key = keys[idx]
                break
        except ValueError:
            pass
        print(f"  invalid choice '{raw}'.")

    picks = _pick_within_template(registry.templates[template_key])
    if picks is None:
        print("Cancelled.")
        return 0
    tokenizer_id, provider_id, metric_override = picks
    save_instance_flow(
        template_key=template_key,
        tokenizer_id=tokenizer_id,
        provider_id=provider_id,
        metric_override=metric_override,
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry = load_registry()
    if args.template_key not in registry.templates:
        print(f"Template '{args.template_key}' not found.")
        return 1
    template = registry.templates[args.template_key]
    report = validate_full_template(template)
    save_registry(registry)
    for c in report.checks:
        marker = "OK" if c.ok else "FAIL"
        print(f"  [{marker}] {c.name}: {c.detail}")
    return 0 if report.ok else 2


def cmd_refresh(_args: argparse.Namespace) -> int:
    registry = load_registry()
    if not registry.templates:
        print("No templates to refresh.")
        return 0
    any_fail = False
    for tk, t in registry.templates.items():
        print(f"\nRefreshing template '{tk}' ...")
        report = validate_full_template(t)
        for c in report.checks:
            marker = "OK" if c.ok else "FAIL"
            print(f"  [{marker}] {c.name}: {c.detail}")
        if not report.ok:
            any_fail = True
    save_registry(registry)
    return 0 if not any_fail else 2


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
        "register-template", help="Interactively register a new template"
    ).set_defaults(func=cmd_register_template)
    subs.add_parser(
        "save-instance", help="Interactively save a named instance from a template"
    ).set_defaults(func=cmd_save_instance)

    p_validate = subs.add_parser(
        "validate", help="Re-validate every tokenizer/provider in one template"
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
