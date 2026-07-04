"""
Dataclasses backing the YAML registry.

The registry models four independent building blocks, each with its own
two-layer template/instance split, plus a composite that ties one of
each together:

* Embedding — `EmbeddingTemplate` (a model's fixed metric + dimension
  plus alternative tokenizers and providers) and `EmbeddingInstanceSpec`
  (a locked tokenizer + provider + optional metric override).

* Reranker — `RerankerTemplate` (a reranker model with alternative
  providers) and `RerankerInstanceSpec` (a locked provider).

* LLM — `LLMTemplate` (an LLM model with alternative providers) and
  `LLMInstanceSpec` (a locked provider).

* Search — a static catalogue of searcher class names (`search_strategies`)
  and `SearchInstanceSpec` (a chosen subset to run).

* `MotherInstanceSpec` — the top-level pick: one embedding instance, one
  search instance, and optionally one reranker instance and one LLM
  instance. This is what the pipeline is parameterised by at runtime.

The registry file holds every catalogue and every named pick in one
document under separate top-level keys.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


KNOWN_SEARCH_STRATEGIES: List[str] = [
    "ANNSearcher",
    "IVFSearcher",
    "LSHSearcher",
    "AnnoySearcher",
    "KNNSearcher",
]


class Status(str, Enum):
    """
    Lifecycle state of a tokenizer or provider entry inside a template.

    `stable`        — last validation passed.
    `unreachable`   — provider network probe failed (e.g. Ollama not running).
    `unavailable`   — provider was reached but it does not serve the model,
                       or the tokenizer repo could not be loaded.
    `not_validated` — has never been checked, or was just freshly added.
    """

    STABLE = "stable"
    UNREACHABLE = "unreachable"
    UNAVAILABLE = "unavailable"
    NOT_VALIDATED = "not_validated"


@dataclass
class TokenizerSpec:
    id: str
    kind: str
    repo: str
    status: Status = Status.NOT_VALIDATED
    last_checked_at: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class ProviderSpec:
    id: str
    kind: str
    model_id: str
    default_base_url: Optional[str] = None
    requires_api_key: bool = False
    status: Status = Status.NOT_VALIDATED
    last_checked_at: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class RerankerSpec:
    """
    A cross-encoder reranker provider.

    Mirrors `ProviderSpec` because rerankers, like embedders, are a
    `(transport, model_id)` pair — the kind discriminates between
    Ollama-served rerankers and locally-loaded HuggingFace
    cross-encoders. `score_strategy` is Ollama-only and chooses between
    `/api/embed` and `/api/generate` scoring.
    """

    id: str
    kind: str
    model_id: str
    default_base_url: Optional[str] = None
    requires_api_key: bool = False
    score_strategy: Optional[str] = None
    status: Status = Status.NOT_VALIDATED
    last_checked_at: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class EmbeddingTemplate:
    model_key: str
    metric: str
    dimension: Optional[int] = None
    tokenizers: Dict[str, TokenizerSpec] = field(default_factory=dict)
    providers: Dict[str, ProviderSpec] = field(default_factory=dict)


@dataclass
class RerankerTemplate:
    """A reranker model with alternative providers to serve it."""

    model_key: str
    providers: Dict[str, RerankerSpec] = field(default_factory=dict)


@dataclass
class LLMTemplate:
    """An LLM model with alternative providers to serve it."""

    model_key: str
    providers: Dict[str, ProviderSpec] = field(default_factory=dict)


@dataclass
class EmbeddingInstanceSpec:
    name: str
    template_key: str
    tokenizer_id: str
    provider_id: str
    metric_override: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class RerankerInstanceSpec:
    name: str
    template_key: str
    provider_id: str
    created_at: Optional[str] = None


@dataclass
class LLMInstanceSpec:
    name: str
    template_key: str
    provider_id: str
    created_at: Optional[str] = None


@dataclass
class SearchInstanceSpec:
    name: str
    strategies: List[str] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class MotherInstanceSpec:
    """
    The top-level pick. References one sub-instance of each kind by name.
    `reranker_instance` and `llm_instance` are optional — a mother
    instance can run embedding + search alone.
    """

    name: str
    embedding_instance: str
    search_instance: str
    reranker_instance: Optional[str] = None
    llm_instance: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Registry:
    schema_version: int = 2
    embedding_templates: Dict[str, EmbeddingTemplate] = field(default_factory=dict)
    reranker_templates: Dict[str, RerankerTemplate] = field(default_factory=dict)
    llm_templates: Dict[str, LLMTemplate] = field(default_factory=dict)
    search_strategies: List[str] = field(
        default_factory=lambda: list(KNOWN_SEARCH_STRATEGIES)
    )
    embedding_instances: Dict[str, EmbeddingInstanceSpec] = field(default_factory=dict)
    reranker_instances: Dict[str, RerankerInstanceSpec] = field(default_factory=dict)
    llm_instances: Dict[str, LLMInstanceSpec] = field(default_factory=dict)
    search_instances: Dict[str, SearchInstanceSpec] = field(default_factory=dict)
    mother_instances: Dict[str, MotherInstanceSpec] = field(default_factory=dict)
