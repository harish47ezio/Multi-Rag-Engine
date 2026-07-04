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
from typing import Any, Dict, List

import yaml

from rag.registry.schema import (
    KNOWN_SEARCH_STRATEGIES,
    EmbeddingInstanceSpec,
    EmbeddingTemplate,
    LLMInstanceSpec,
    LLMTemplate,
    MotherInstanceSpec,
    ProviderSpec,
    Registry,
    RerankerInstanceSpec,
    RerankerSpec,
    RerankerTemplate,
    SearchInstanceSpec,
    Status,
    TokenizerSpec,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(os.environ.get("MULTI_RAG_REGISTRY", "registry.yaml"))


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    if not path.exists():
        logger.info("load_registry no file at path=%s (returning empty)", path)
        return Registry(schema_version=2)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    schema_version = int(data.get("schema_version", 2))

    embedding_templates: Dict[str, EmbeddingTemplate] = {}
    for model_key, t in (data.get("embedding_templates") or {}).items():
        embedding_templates[model_key] = EmbeddingTemplate(
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
                pid: _provider_from_dict(pid, spec)
                for pid, spec in (t.get("providers") or {}).items()
            },
        )

    reranker_templates: Dict[str, RerankerTemplate] = {}
    for model_key, t in (data.get("reranker_templates") or {}).items():
        reranker_templates[model_key] = RerankerTemplate(
            model_key=model_key,
            providers={
                pid: _reranker_from_dict(pid, spec)
                for pid, spec in (t.get("providers") or {}).items()
            },
        )

    llm_templates: Dict[str, LLMTemplate] = {}
    for model_key, t in (data.get("llm_templates") or {}).items():
        llm_templates[model_key] = LLMTemplate(
            model_key=model_key,
            providers={
                pid: _provider_from_dict(pid, spec)
                for pid, spec in (t.get("providers") or {}).items()
            },
        )

    search_strategies: List[str] = list(
        data.get("search_strategies") or KNOWN_SEARCH_STRATEGIES
    )

    embedding_instances: Dict[str, EmbeddingInstanceSpec] = {}
    for name, i in (data.get("embedding_instances") or {}).items():
        embedding_instances[name] = EmbeddingInstanceSpec(
            name=name,
            template_key=str(i["template_key"]),
            tokenizer_id=str(i["tokenizer_id"]),
            provider_id=str(i["provider_id"]),
            metric_override=i.get("metric_override"),
            created_at=i.get("created_at"),
        )

    reranker_instances: Dict[str, RerankerInstanceSpec] = {}
    for name, i in (data.get("reranker_instances") or {}).items():
        reranker_instances[name] = RerankerInstanceSpec(
            name=name,
            template_key=str(i["template_key"]),
            provider_id=str(i["provider_id"]),
            created_at=i.get("created_at"),
        )

    llm_instances: Dict[str, LLMInstanceSpec] = {}
    for name, i in (data.get("llm_instances") or {}).items():
        llm_instances[name] = LLMInstanceSpec(
            name=name,
            template_key=str(i["template_key"]),
            provider_id=str(i["provider_id"]),
            created_at=i.get("created_at"),
        )

    search_instances: Dict[str, SearchInstanceSpec] = {}
    for name, i in (data.get("search_instances") or {}).items():
        search_instances[name] = SearchInstanceSpec(
            name=name,
            strategies=list(i.get("strategies") or []),
            created_at=i.get("created_at"),
        )

    mother_instances: Dict[str, MotherInstanceSpec] = {}
    for name, i in (data.get("mother_instances") or {}).items():
        mother_instances[name] = MotherInstanceSpec(
            name=name,
            embedding_instance=str(i["embedding_instance"]),
            search_instance=str(i["search_instance"]),
            reranker_instance=i.get("reranker_instance"),
            llm_instance=i.get("llm_instance"),
            created_at=i.get("created_at"),
        )

    logger.info(
        "load_registry loaded path=%s embedding_templates=%d reranker_templates=%d "
        "llm_templates=%d mother_instances=%d",
        path,
        len(embedding_templates),
        len(reranker_templates),
        len(llm_templates),
        len(mother_instances),
    )
    return Registry(
        schema_version=schema_version,
        embedding_templates=embedding_templates,
        reranker_templates=reranker_templates,
        llm_templates=llm_templates,
        search_strategies=search_strategies,
        embedding_instances=embedding_instances,
        reranker_instances=reranker_instances,
        llm_instances=llm_instances,
        search_instances=search_instances,
        mother_instances=mother_instances,
    )


def save_registry(registry: Registry, path: Path = REGISTRY_PATH) -> None:
    """Serialize to YAML atomically (write-temp + rename)."""
    payload: Dict[str, Any] = {
        "schema_version": registry.schema_version,
        "embedding_templates": {
            tk: {
                "metric": t.metric,
                "dimension": t.dimension,
                "tokenizers": {
                    tid: _tokenizer_to_dict(s) for tid, s in t.tokenizers.items()
                },
                "providers": {
                    pid: _provider_to_dict(s) for pid, s in t.providers.items()
                },
            }
            for tk, t in registry.embedding_templates.items()
        },
        "reranker_templates": {
            tk: {
                "providers": {
                    pid: _reranker_to_dict(s) for pid, s in t.providers.items()
                },
            }
            for tk, t in registry.reranker_templates.items()
        },
        "llm_templates": {
            tk: {
                "providers": {
                    pid: _provider_to_dict(s) for pid, s in t.providers.items()
                },
            }
            for tk, t in registry.llm_templates.items()
        },
        "search_strategies": list(registry.search_strategies),
        "embedding_instances": {
            name: {
                "template_key": i.template_key,
                "tokenizer_id": i.tokenizer_id,
                "provider_id": i.provider_id,
                "metric_override": i.metric_override,
                "created_at": i.created_at,
            }
            for name, i in registry.embedding_instances.items()
        },
        "reranker_instances": {
            name: {
                "template_key": i.template_key,
                "provider_id": i.provider_id,
                "created_at": i.created_at,
            }
            for name, i in registry.reranker_instances.items()
        },
        "llm_instances": {
            name: {
                "template_key": i.template_key,
                "provider_id": i.provider_id,
                "created_at": i.created_at,
            }
            for name, i in registry.llm_instances.items()
        },
        "search_instances": {
            name: {
                "strategies": list(i.strategies),
                "created_at": i.created_at,
            }
            for name, i in registry.search_instances.items()
        },
        "mother_instances": {
            name: {
                "embedding_instance": i.embedding_instance,
                "search_instance": i.search_instance,
                "reranker_instance": i.reranker_instance,
                "llm_instance": i.llm_instance,
                "created_at": i.created_at,
            }
            for name, i in registry.mother_instances.items()
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


def _provider_from_dict(pid: str, spec: Dict[str, Any]) -> ProviderSpec:
    return ProviderSpec(
        id=pid,
        kind=str(spec.get("kind", "ollama")),
        model_id=str(spec.get("model_id", "")),
        default_base_url=spec.get("default_base_url"),
        requires_api_key=bool(spec.get("requires_api_key", False)),
        status=_status(spec.get("status")),
        last_checked_at=spec.get("last_checked_at"),
        detail=spec.get("detail"),
    )


def _reranker_from_dict(pid: str, spec: Dict[str, Any]) -> RerankerSpec:
    return RerankerSpec(
        id=pid,
        kind=str(spec.get("kind", "ollama")),
        model_id=str(spec.get("model_id", "")),
        default_base_url=spec.get("default_base_url"),
        requires_api_key=bool(spec.get("requires_api_key", False)),
        score_strategy=spec.get("score_strategy"),
        status=_status(spec.get("status")),
        last_checked_at=spec.get("last_checked_at"),
        detail=spec.get("detail"),
    )


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
