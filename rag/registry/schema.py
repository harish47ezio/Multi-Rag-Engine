"""
Dataclasses backing the YAML registry.

Two top-level concepts:

* `Template` — a model definition with N tokenizer entries and N provider
  entries. The metric and dimension are fixed properties of the model;
  the tokenizer and provider lists are alternatives the user is free to
  pick between. Each entry carries its own `status` and `last_checked_at`
  so the wizard can save partially-broken templates and have the user
  fix them later.

* `SavedInstance` — a named, locked pick (template + one tokenizer + one
  provider + optional metric override). Instances are how the pipeline
  is actually parameterised at runtime; templates only describe what
  picks are *available*.

The registry file holds both in one document under separate top-level
keys: `templates:` and `instances:`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


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
    Optional cross-encoder reranker attached to a template.

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
class Template:
    model_key: str
    metric: str
    dimension: Optional[int] = None
    tokenizers: Dict[str, TokenizerSpec] = field(default_factory=dict)
    providers: Dict[str, ProviderSpec] = field(default_factory=dict)
    rerankers: Dict[str, RerankerSpec] = field(default_factory=dict)


@dataclass
class SavedInstance:
    name: str
    template_key: str
    tokenizer_id: str
    provider_id: str
    metric_override: Optional[str] = None
    reranker_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Registry:
    schema_version: int = 1
    templates: Dict[str, Template] = field(default_factory=dict)
    instances: Dict[str, SavedInstance] = field(default_factory=dict)
