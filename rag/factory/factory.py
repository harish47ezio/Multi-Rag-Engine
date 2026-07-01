"""
Build an `Instance` from the registry.

Three production entrypoints:

  * `from_template(model_key, tokenizer_id?, provider_id?, ...)` — pick a
    template by canonical key and a specific tokenizer/provider within
    it. Used programmatically (tests, library code).

  * `from_saved_instance(name)` — re-hydrate a previously-saved
    `SavedInstance` (saved in `registry.yaml` under `instances:`).

  * `interactive_pick()` — walk the user through the CLI picker. Used by
    alpha_test scripts and any human-facing entrypoint.

Validation is lazy: only the picks actually used here are checked, and
the touched template (with refreshed `status` / `last_checked_at`) is
persisted back to disk before returning the Instance.

The factory composes an Instance from three orthogonal parts —
`BaseTokenizer`, `BaseEmbedder`, `BaseDistanceMetric` — each built by
its own small helper that dispatches on the spec's `kind`. Adding a
new tokenizer or provider kind = one new arm in the matching helper,
nothing else.
"""

from __future__ import annotations

import logging
from typing import Optional

from providers.ollama_client import OllamaClient
from providers.hugging_face_client import HuggingFaceClient
from rag.embedder.base_embedder import BaseEmbedder
from rag.embedder.hugging_face_adapter import HuggingFaceAdapter
from rag.embedder.ollama_adapter import OllamaAdapter
from rag.factory.instance import Instance
from rag.registry.loader import REGISTRY_PATH, load_registry, save_registry
from rag.registry.schema import ProviderSpec, RerankerSpec, TokenizerSpec
from rag.registry.validator import validate_template_picks
from rag.reranker.base_reranker import BaseReranker
from rag.reranker.ollama_reranker import OllamaReranker
from rag.search.distance_metrics import build_metric
from rag.search.distance_metrics.base_distance_metric import MetricKind
from rag.tokenizer.base_tokenizer import BaseTokenizer
from rag.tokenizer.hf_tokenizer import HFTokenizer

logger = logging.getLogger(__name__)


class EmbedderFactory:

    @staticmethod
    def from_template(
        model_key: str,
        tokenizer_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metric_override: Optional[str] = None,
        reranker_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Instance:
        """
        Build an `Instance` from a template in the registry.

        If `tokenizer_id` or `provider_id` is None, the first entry in
        the template's respective dict is used (insertion order from
        YAML). Pass `metric_override` only for explicit experiments —
        it logs a warning because it diverges from the model's training
        objective. `reranker_id` is fully optional: when None and the
        template defines no rerankers, the resulting Instance simply
        has `reranker=None` and the pipeline skips reranking.
        """
        registry = load_registry()
        if model_key not in registry.templates:
            raise ValueError(
                f"Template '{model_key}' not found in registry at {REGISTRY_PATH}. "
                f"Run `python -m rag.registry register-template` to add it."
            )
        template = registry.templates[model_key]

        if tokenizer_id is None:
            if not template.tokenizers:
                raise ValueError(
                    f"Template '{model_key}' has no tokenizer entries; "
                    f"add one via `python -m rag.registry register-template`."
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
                    f"Template '{model_key}' has no provider entries; "
                    f"add one via `python -m rag.registry register-template`."
                )
            provider_id = next(iter(template.providers))
        if provider_id not in template.providers:
            raise ValueError(
                f"Provider '{provider_id}' not found in template '{model_key}'. "
                f"Available: {list(template.providers)}"
            )

        if reranker_id is not None and reranker_id not in template.rerankers:
            raise ValueError(
                f"Reranker '{reranker_id}' not found in template '{model_key}'. "
                f"Available: {list(template.rerankers) or '[]'}"
            )

        report = validate_template_picks(
            template, tokenizer_id, provider_id, reranker_id=reranker_id
        )
        save_registry(registry)

        if not report.ok:
            logger.warning(
                "from_template validation issues for %s / %s / %s — proceeding anyway:",
                model_key,
                tokenizer_id,
                provider_id,
            )
            for c in report.checks:
                marker = "OK" if c.ok else "FAIL"
                logger.warning("  [%s] %s: %s", marker, c.name, c.detail)
        else:
            logger.info(
                "from_template validated %s / %s / %s OK",
                model_key,
                tokenizer_id,
                provider_id,
            )

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

        tokenizer = _build_tokenizer(tok_spec)
        embedder = _build_embedder(prov_spec, metric_kind, api_key=api_key)
        metric_obj = build_metric(metric_kind)

        reranker: Optional[BaseReranker] = None
        if reranker_id is not None:
            reranker = _build_reranker(
                template.rerankers[reranker_id], api_key=api_key
            )

        return Instance(
            template_key=model_key,
            model_key=template.model_key,
            tokenizer_id=tokenizer_id,
            provider_id=provider_id,
            tokenizer=tokenizer,
            embedder=embedder,
            metric=metric_obj,
            reranker_id=reranker_id,
            reranker=reranker,
        )

    @staticmethod
    def from_saved_instance(
        name: str,
        api_key: Optional[str] = None,
    ) -> Instance:
        registry = load_registry()
        if name not in registry.instances:
            raise ValueError(
                f"Saved instance '{name}' not found in registry at {REGISTRY_PATH}."
            )
        saved = registry.instances[name]
        logger.info(
            "from_saved_instance name=%s template=%s tokenizer=%s provider=%s",
            name,
            saved.template_key,
            saved.tokenizer_id,
            saved.provider_id,
        )
        return EmbedderFactory.from_template(
            model_key=saved.template_key,
            tokenizer_id=saved.tokenizer_id,
            provider_id=saved.provider_id,
            metric_override=saved.metric_override,
            reranker_id=saved.reranker_id,
            api_key=api_key,
        )

    @staticmethod
    def interactive_pick() -> Instance:
        from rag.registry.picker import run_picker

        return run_picker()


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
