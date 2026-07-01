"""
Load and save the registry YAML.

Reads `registry.yaml` (location overridable via `MULTI_RAG_REGISTRY` env
var). If the file is missing, returns an empty `Registry` rather than
erroring — the picker can then offer the "register new template" flow
right away.

Saving is atomic (write-to-tempfile + rename) so a crash mid-write can
never corrupt the live registry. Field order in the on-disk document
matches the schema's logical order for readability of hand-edits.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from rag.registry.schema import (
    ProviderSpec,
    Registry,
    RerankerSpec,
    SavedInstance,
    Status,
    Template,
    TokenizerSpec,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(os.environ.get("MULTI_RAG_REGISTRY", "registry.yaml"))


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    if not path.exists():
        logger.info("load_registry no file at path=%s (returning empty)", path)
        return Registry(schema_version=1)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    schema_version = int(data.get("schema_version", 1))

    templates: Dict[str, Template] = {}
    for model_key, t in (data.get("templates") or {}).items():
        templates[model_key] = Template(
            model_key=model_key,
            metric=str(t.get("metric", "cosine")),
            dimension=t.get("dimension"),
            tokenizers={
                tid: TokenizerSpec(
                    id=tid,
                    kind=str(spec.get("kind", "hf")),
                    repo=str(spec.get("repo", "")),
                    status=_status(spec.get("status")),
                    last_checked_at=spec.get("last_checked_at"),
                    detail=spec.get("detail"),
                )
                for tid, spec in (t.get("tokenizers") or {}).items()
            },
            providers={
                pid: ProviderSpec(
                    id=pid,
                    kind=str(spec.get("kind", "ollama")),
                    model_id=str(spec.get("model_id", "")),
                    default_base_url=spec.get("default_base_url"),
                    requires_api_key=bool(spec.get("requires_api_key", False)),
                    status=_status(spec.get("status")),
                    last_checked_at=spec.get("last_checked_at"),
                    detail=spec.get("detail"),
                )
                for pid, spec in (t.get("providers") or {}).items()
            },
            rerankers={
                rid: RerankerSpec(
                    id=rid,
                    kind=str(spec.get("kind", "ollama")),
                    model_id=str(spec.get("model_id", "")),
                    default_base_url=spec.get("default_base_url"),
                    requires_api_key=bool(spec.get("requires_api_key", False)),
                    score_strategy=spec.get("score_strategy"),
                    status=_status(spec.get("status")),
                    last_checked_at=spec.get("last_checked_at"),
                    detail=spec.get("detail"),
                )
                for rid, spec in (t.get("rerankers") or {}).items()
            },
        )

    instances: Dict[str, SavedInstance] = {}
    for name, i in (data.get("instances") or {}).items():
        instances[name] = SavedInstance(
            name=name,
            template_key=str(i["template_key"]),
            tokenizer_id=str(i["tokenizer_id"]),
            provider_id=str(i["provider_id"]),
            metric_override=i.get("metric_override"),
            reranker_id=i.get("reranker_id"),
            created_at=i.get("created_at"),
        )

    logger.info(
        "load_registry loaded path=%s templates=%d instances=%d",
        path,
        len(templates),
        len(instances),
    )
    return Registry(schema_version=schema_version, templates=templates, instances=instances)


def save_registry(registry: Registry, path: Path = REGISTRY_PATH) -> None:
    """Serialize to YAML atomically (write-temp + rename)."""
    payload: Dict[str, Any] = {
        "schema_version": registry.schema_version,
        "templates": {
            tk: {
                "metric": t.metric,
                "dimension": t.dimension,
                "tokenizers": {
                    tid: _tokenizer_to_dict(s) for tid, s in t.tokenizers.items()
                },
                "providers": {
                    pid: _provider_to_dict(s) for pid, s in t.providers.items()
                },
                "rerankers": {
                    rid: _reranker_to_dict(s) for rid, s in t.rerankers.items()
                },
            }
            for tk, t in registry.templates.items()
        },
        "instances": {
            name: {
                "template_key": i.template_key,
                "tokenizer_id": i.tokenizer_id,
                "provider_id": i.provider_id,
                "metric_override": i.metric_override,
                "reranker_id": i.reranker_id,
                "created_at": i.created_at,
            }
            for name, i in registry.instances.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".registry-", suffix=".yaml", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp_path, path)
        logger.info("save_registry written path=%s", path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _status(raw: Any) -> Status:
    try:
        return Status(str(raw)) if raw is not None else Status.NOT_VALIDATED
    except ValueError:
        return Status.NOT_VALIDATED


def _tokenizer_to_dict(s: TokenizerSpec) -> Dict[str, Any]:
    return {
        "kind": s.kind,
        "repo": s.repo,
        "status": s.status.value,
        "last_checked_at": s.last_checked_at,
        "detail": s.detail,
    }


def _provider_to_dict(s: ProviderSpec) -> Dict[str, Any]:
    return {
        "kind": s.kind,
        "model_id": s.model_id,
        "default_base_url": s.default_base_url,
        "requires_api_key": s.requires_api_key,
        "status": s.status.value,
        "last_checked_at": s.last_checked_at,
        "detail": s.detail,
    }


def _reranker_to_dict(s: RerankerSpec) -> Dict[str, Any]:
    return {
        "kind": s.kind,
        "model_id": s.model_id,
        "default_base_url": s.default_base_url,
        "requires_api_key": s.requires_api_key,
        "score_strategy": s.score_strategy,
        "status": s.status.value,
        "last_checked_at": s.last_checked_at,
        "detail": s.detail,
    }
