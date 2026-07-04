"""
Build runtime instances from the registry.

One factory per building block, plus a composite:

  * `EmbeddingFactory` — `from_template(...)` / `from_saved_instance(name)`
    -> `EmbeddingInstance`.
  * `RerankerFactory`  — `from_template(...)` / `from_saved_instance(name)`
    -> `RerankerInstance`.
  * `LLMFactory`       — `from_template(...)` / `from_saved_instance(name)`
    -> `LLMInstance`.
  * `MotherFactory`    — `from_saved(name)` / `interactive_pick()`
    -> `MotherInstance` (embedding + search + optional reranker + optional llm).

Validation is lazy: only the picks actually used are checked, and the
touched template (with refreshed `status` / `last_checked_at`) is
persisted back to disk before returning.

Each building block composes concrete parts via a small helper that
dispatches on the spec's `kind`. Adding a new tokenizer/provider/reranker
kind = one new arm in the matching helper, nothing else.
"""

from __future__ import annotations

import logging
from typing import Optional

from llm.factory import LLMFactory as LLMAdapterFactory
from providers.hugging_face_client import HuggingFaceClient
from providers.ollama_client import OllamaClient
from rag.embedder.base_embedder import BaseEmbedder
from rag.embedder.hugging_face_adapter import HuggingFaceAdapter
from rag.embedder.ollama_adapter import OllamaAdapter
from rag.factory.instance import (
    EmbeddingInstance,
    LLMInstance,
    MotherInstance,
    RerankerInstance,
    SearchInstance,
)
from rag.registry.loader import REGISTRY_PATH, load_registry, save_registry
from rag.registry.schema import (
    LLMInstanceSpec,
    ProviderSpec,
    Registry,
    RerankerSpec,
    SearchInstanceSpec,
    TokenizerSpec,
)
from rag.registry.validator import (
    validate_embedding_picks,
    validate_llm_pick,
    validate_reranker_pick,
)
from rag.reranker.base_reranker import BaseReranker
from rag.reranker.ollama_reranker import OllamaReranker
from rag.search.distance_metrics import build_metric
from rag.search.distance_metrics.base_distance_metric import MetricKind
from rag.tokenizer.base_tokenizer import BaseTokenizer
from rag.tokenizer.hf_tokenizer import HFTokenizer

logger = logging.getLogger(__name__)


class EmbeddingFactory:

    @staticmethod
    def from_template(
        model_key: str,
        tokenizer_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metric_override: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> EmbeddingInstance:
        """
        Build an `EmbeddingInstance` from an embedding template.

        If `tokenizer_id` / `provider_id` is None, the first entry in the
        template's respective dict is used (YAML insertion order). Pass
        `metric_override` only for explicit experiments — it logs a
        warning because it diverges from the model's training objective.
        """
        registry = load_registry()
        if model_key not in registry.embedding_templates:
            raise ValueError(
                f"Embedding template '{model_key}' not found in registry at "
                f"{REGISTRY_PATH}. Run `python -m rag.registry register-template`."
            )
        template = registry.embedding_templates[model_key]

        if tokenizer_id is None:
            if not template.tokenizers:
                raise ValueError(
                    f"Embedding template '{model_key}' has no tokenizer entries."
                )
            tokenizer_id = next(iter(template.tokenizers))
        if tokenizer_id not in template.tokenizers:
            raise ValueError(
                f"Tokenizer '{tokenizer_id}' not found in template '{model_key}'. "
                f"Available: {list(template.tokenizers)}"
            )

        if provider_id is None:
            if not template.providers:
                raise ValueError(
                    f"Embedding template '{model_key}' has no provider entries."
                )
            provider_id = next(iter(template.providers))
        if provider_id not in template.providers:
            raise ValueError(
                f"Provider '{provider_id}' not found in template '{model_key}'. "
                f"Available: {list(template.providers)}"
            )

        report = validate_embedding_picks(template, tokenizer_id, provider_id)
        save_registry(registry)
        _log_report("embedding", model_key, report, f"{tokenizer_id}/{provider_id}")

        tok_spec = template.tokenizers[tokenizer_id]
        prov_spec = template.providers[provider_id]
        metric_str = metric_override or template.metric
        if metric_override is not None and metric_override != template.metric:
            logger.warning(
                "Metric override active: template default=%s, using=%s. "
                "Retrieval quality may differ from what the model was trained for.",
                template.metric,
                metric_override,
            )
        try:
            metric_kind = MetricKind(metric_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid metric '{metric_str}' for template '{model_key}'. "
                f"Choose from: {[m.value for m in MetricKind]}"
            ) from exc

        return EmbeddingInstance(
            name=name,
            template_key=model_key,
            model_key=template.model_key,
            tokenizer_id=tokenizer_id,
            provider_id=provider_id,
            tokenizer=_build_tokenizer(tok_spec),
            embedder=_build_embedder(prov_spec, metric_kind, api_key=api_key),
            metric=build_metric(metric_kind),
        )

    @staticmethod
    def from_saved_instance(
        name: str,
        api_key: Optional[str] = None,
    ) -> EmbeddingInstance:
        registry = load_registry()
        if name not in registry.embedding_instances:
            raise ValueError(
                f"Embedding instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        saved = registry.embedding_instances[name]
        logger.info(
            "EmbeddingFactory.from_saved_instance name=%s template=%s tokenizer=%s provider=%s",
            name,
            saved.template_key,
            saved.tokenizer_id,
            saved.provider_id,
        )
        return EmbeddingFactory.from_template(
            model_key=saved.template_key,
            tokenizer_id=saved.tokenizer_id,
            provider_id=saved.provider_id,
            metric_override=saved.metric_override,
            api_key=api_key,
            name=name,
        )


class RerankerFactory:

    @staticmethod
    def from_template(
        model_key: str,
        provider_id: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> RerankerInstance:
        registry = load_registry()
        if model_key not in registry.reranker_templates:
            raise ValueError(
                f"Reranker template '{model_key}' not found in registry at "
                f"{REGISTRY_PATH}. Run `python -m rag.registry register-reranker`."
            )
        template = registry.reranker_templates[model_key]

        if provider_id is None:
            if not template.providers:
                raise ValueError(
                    f"Reranker template '{model_key}' has no provider entries."
                )
            provider_id = next(iter(template.providers))
        if provider_id not in template.providers:
            raise ValueError(
                f"Provider '{provider_id}' not found in reranker template '{model_key}'. "
                f"Available: {list(template.providers)}"
            )

        report = validate_reranker_pick(template, provider_id)
        save_registry(registry)
        _log_report("reranker", model_key, report, provider_id)

        return RerankerInstance(
            name=name,
            template_key=model_key,
            provider_id=provider_id,
            reranker=_build_reranker(template.providers[provider_id], api_key=api_key),
        )

    @staticmethod
    def from_saved_instance(
        name: str,
        api_key: Optional[str] = None,
    ) -> RerankerInstance:
        registry = load_registry()
        if name not in registry.reranker_instances:
            raise ValueError(
                f"Reranker instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        saved = registry.reranker_instances[name]
        return RerankerFactory.from_template(
            model_key=saved.template_key,
            provider_id=saved.provider_id,
            api_key=api_key,
            name=name,
        )


class LLMFactory:

    @staticmethod
    def from_template(
        model_key: str,
        provider_id: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> LLMInstance:
        registry = load_registry()
        if model_key not in registry.llm_templates:
            raise ValueError(
                f"LLM template '{model_key}' not found in registry at "
                f"{REGISTRY_PATH}. Run `python -m rag.registry register-llm`."
            )
        template = registry.llm_templates[model_key]

        if provider_id is None:
            if not template.providers:
                raise ValueError(
                    f"LLM template '{model_key}' has no provider entries."
                )
            provider_id = next(iter(template.providers))
        if provider_id not in template.providers:
            raise ValueError(
                f"Provider '{provider_id}' not found in LLM template '{model_key}'. "
                f"Available: {list(template.providers)}"
            )

        report = validate_llm_pick(template, provider_id)
        save_registry(registry)
        _log_report("llm", model_key, report, provider_id)

        return LLMInstance(
            name=name,
            template_key=model_key,
            provider_id=provider_id,
            llm=_build_llm(template.providers[provider_id], api_key=api_key),
        )

    @staticmethod
    def from_saved_instance(
        name: str,
        api_key: Optional[str] = None,
    ) -> LLMInstance:
        registry = load_registry()
        if name not in registry.llm_instances:
            raise ValueError(
                f"LLM instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        saved = registry.llm_instances[name]
        return LLMFactory.from_template(
            model_key=saved.template_key,
            provider_id=saved.provider_id,
            api_key=api_key,
            name=name,
        )


class MotherFactory:

    @staticmethod
    def from_saved(name: str, api_key: Optional[str] = None) -> MotherInstance:
        registry = load_registry()
        if name not in registry.mother_instances:
            raise ValueError(
                f"Mother instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        spec = registry.mother_instances[name]
        logger.info(
            "MotherFactory.from_saved name=%s embedding=%s search=%s reranker=%s llm=%s",
            name,
            spec.embedding_instance,
            spec.search_instance,
            spec.reranker_instance,
            spec.llm_instance,
        )

        embedding = EmbeddingFactory.from_saved_instance(
            spec.embedding_instance, api_key=api_key
        )
        search = MotherFactory.build_search(registry, spec.search_instance)
        reranker = (
            RerankerFactory.from_saved_instance(spec.reranker_instance, api_key=api_key)
            if spec.reranker_instance
            else None
        )
        llm = (
            LLMFactory.from_saved_instance(spec.llm_instance, api_key=api_key)
            if spec.llm_instance
            else None
        )
        return MotherInstance(
            name=name,
            embedding=embedding,
            search=search,
            reranker=reranker,
            llm=llm,
        )

    @staticmethod
    def build_search(registry: Registry, name: str) -> SearchInstance:
        if name not in registry.search_instances:
            raise ValueError(
                f"Search instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        spec: SearchInstanceSpec = registry.search_instances[name]
        return SearchInstance(name=name, strategies=list(spec.strategies))

    @staticmethod
    def interactive_pick() -> MotherInstance:
        from rag.registry.picker import run_picker

        return run_picker()


def _log_report(kind: str, model_key: str, report, picks: str) -> None:
    if not report.ok:
        logger.warning(
            "%s validation issues for %s (%s) — proceeding anyway:",
            kind,
            model_key,
            picks,
        )
        for c in report.checks:
            marker = "OK" if c.ok else "FAIL"
            logger.warning("  [%s] %s: %s", marker, c.name, c.detail)
    else:
        logger.info("%s validated %s (%s) OK", kind, model_key, picks)


def _build_tokenizer(spec: TokenizerSpec) -> BaseTokenizer:
    """Dispatch on `spec.kind` to build the right `BaseTokenizer`."""
    if spec.kind == "hf":
        return HFTokenizer(repo=spec.repo)
    raise ValueError(
        f"Unknown tokenizer kind '{spec.kind}'. Supported kinds today: ['hf']."
    )


def _build_embedder(
    provider_spec: ProviderSpec,
    metric_kind: MetricKind,
    api_key: Optional[str] = None,
) -> BaseEmbedder:
    """Dispatch on `provider_spec.kind` to build the right `BaseEmbedder` (transport)."""
    if provider_spec.kind == "ollama":
        client = OllamaClient(
            base_url=provider_spec.default_base_url,
            api_key=api_key,
        )
        return OllamaAdapter(
            client=client,
            model=provider_spec.model_id,
            metric_kind=metric_kind,
        )
    if provider_spec.kind == "hf":
        client = HuggingFaceClient(
            base_url=provider_spec.default_base_url,
            api_key=api_key,
        )
        return HuggingFaceAdapter(
            client=client,
            model=provider_spec.model_id,
            metric_kind=metric_kind,
        )
    raise ValueError(
        f"Unknown provider kind '{provider_spec.kind}'. "
        f"Supported kinds today: ['ollama', 'hf']."
    )


def _build_reranker(
    reranker_spec: RerankerSpec,
    api_key: Optional[str] = None,
) -> BaseReranker:
    """Dispatch on `reranker_spec.kind` to build the right `BaseReranker`."""
    if reranker_spec.kind == "ollama":
        client = OllamaClient(
            base_url=reranker_spec.default_base_url,
            api_key=api_key,
        )
        return OllamaReranker(
            client=client,
            model=reranker_spec.model_id,
            score_strategy=reranker_spec.score_strategy or "embed",
        )
    if reranker_spec.kind == "hf":
        # Deferred import: keeps torch off the path for Ollama-only setups.
        from rag.reranker.hf_reranker import HFReranker

        return HFReranker(repo=reranker_spec.model_id)
    raise ValueError(
        f"Unknown reranker kind '{reranker_spec.kind}'. "
        f"Supported kinds today: ['ollama', 'hf']."
    )


def _build_llm(provider_spec: ProviderSpec, api_key: Optional[str] = None):
    """Dispatch on `provider_spec.kind` to build the right LLM adapter."""
    return LLMAdapterFactory.create(
        {
            "provider": provider_spec.kind,
            "model": provider_spec.model_id,
            "base_url": provider_spec.default_base_url,
            "api_key": api_key,
        }
    )
