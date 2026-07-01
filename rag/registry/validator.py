"""
Validation logic for tokenizer and provider entries.

Three independent checks, each producing its own `CheckResult`:

  1. Tokenizer load    — can we construct `HFTokenizer(repo)`?
  2. Provider reach    — is the transport endpoint up?
  3. Provider serves   — does the endpoint actually expose this model id?

Checks 2 and 3 are folded into `validate_provider()` so callers get one
result per spec; the detail string distinguishes "unreachable" from
"reachable but model not served" so `update_status_from_check` can pick
the right `Status` value.

Validation runs lazily — only when the user picks a (tokenizer,
provider) pair to use, or when they explicitly ask via the CLI. We do
NOT walk every entry on app start.

`validate_tokenizer` constructs the actual concrete `BaseTokenizer`
subclass (e.g. `HFTokenizer`) rather than re-implementing the load
check inline. "Constructible == valid" — one definition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Union

from providers.ollama_client import OllamaClient
from providers.hugging_face_client import HuggingFaceClient
from rag.registry.schema import (
    ProviderSpec,
    RerankerSpec,
    Status,
    Template,
    TokenizerSpec,
)
from rag.tokenizer.hf_tokenizer import HFTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_tokenizer(spec: TokenizerSpec) -> CheckResult:
    """
    Validation by construction: if the concrete `BaseTokenizer` for this
    spec instantiates without raising, the entry is valid. This keeps
    "what we validate" and "what we actually run" in lock-step — no
    chance of validation passing while runtime construction fails.
    """
    if spec.kind == "hf":
        try:
            HFTokenizer(repo=spec.repo)
        except Exception as exc:
            logger.info(
                "validate_tokenizer failed id=%s repo=%s err=%s",
                spec.id,
                spec.repo,
                exc,
            )
            return CheckResult(
                name=f"tokenizer:{spec.id}",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        logger.info("validate_tokenizer ok id=%s repo=%s", spec.id, spec.repo)
        return CheckResult(
            name=f"tokenizer:{spec.id}",
            ok=True,
            detail=f"loaded {spec.repo}",
        )
    return CheckResult(
        name=f"tokenizer:{spec.id}",
        ok=False,
        detail=f"unknown tokenizer kind '{spec.kind}'",
    )


def validate_provider(spec: ProviderSpec) -> CheckResult:
    if spec.kind == "ollama":
        client = OllamaClient(base_url=spec.default_base_url)
        if not client.is_available():
            return CheckResult(
                name=f"provider:{spec.id}",
                ok=False,
                detail=f"unreachable: ollama @ {client.base_url}",
            )
        try:
            served = client.list_model_names()
            if spec.model_id not in served:
                return CheckResult(
                    name=f"provider:{spec.id}",
                    ok=False,
                    detail=(
                        f"model '{spec.model_id}' not served by ollama @ "
                        f"{client.base_url}; available: {sorted(n for n in served if n)[:5]}"
                    ),
                )
            return CheckResult(
                name=f"provider:{spec.id}",
                ok=True,
                detail=f"ollama @ {client.base_url} serving {spec.model_id}",
            )
        except Exception as exc:
            logger.info(
                "validate_provider failed id=%s err=%s",
                spec.id,
                exc,
            )
            return CheckResult(
                name=f"provider:{spec.id}",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
    if spec.kind == "hf":
        client = HuggingFaceClient(base_url=spec.default_base_url)
        if not client.is_available():
            return CheckResult(
                name=f"provider:{spec.id}",
                ok=False,
                detail=f"unreachable: huggingface @ {client.base_url}",
            )
        else:
            return CheckResult(
                name=f"provider:{spec.id}",
                ok=True,
                detail=f"huggingface @ {client.base_url} serving {spec.model_id}",
            )

    return CheckResult(
        name=f"provider:{spec.id}",
        ok=False,
        detail=f"unknown provider kind '{spec.kind}'",
    )


def validate_reranker(spec: RerankerSpec) -> CheckResult:
    """
    Reranker validation, by-construction where possible.

    For Ollama-served rerankers we reuse the same reachability +
    "model is in /api/tags" probe used for embedders — the transport
    constraint is identical. For HF rerankers we attempt to instantiate
    `CrossEncoder(repo)` (deferred import to avoid pulling torch when
    no HF reranker is in play).
    """
    if spec.kind == "ollama":
        client = OllamaClient(base_url=spec.default_base_url)
        if not client.is_available():
            return CheckResult(
                name=f"reranker:{spec.id}",
                ok=False,
                detail=f"unreachable: ollama @ {client.base_url}",
            )
        try:
            served = client.list_model_names()
            if spec.model_id not in served:
                return CheckResult(
                    name=f"reranker:{spec.id}",
                    ok=False,
                    detail=(
                        f"model '{spec.model_id}' not served by ollama @ "
                        f"{client.base_url}; available: {sorted(n for n in served if n)[:5]}"
                    ),
                )
            return CheckResult(
                name=f"reranker:{spec.id}",
                ok=True,
                detail=(
                    f"ollama @ {client.base_url} serving {spec.model_id}"
                    f" (strategy={spec.score_strategy or 'embed'})"
                ),
            )
        except Exception as exc:
            logger.info(
                "validate_reranker ollama failed id=%s err=%s",
                spec.id,
                exc,
            )
            return CheckResult(
                name=f"reranker:{spec.id}",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    if spec.kind == "hf":
        try:
            from sentence_transformers import CrossEncoder

            CrossEncoder(spec.model_id)
        except Exception as exc:
            logger.info(
                "validate_reranker hf failed id=%s repo=%s err=%s",
                spec.id,
                spec.model_id,
                exc,
            )
            return CheckResult(
                name=f"reranker:{spec.id}",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        return CheckResult(
            name=f"reranker:{spec.id}",
            ok=True,
            detail=f"loaded {spec.model_id}",
        )

    return CheckResult(
        name=f"reranker:{spec.id}",
        ok=False,
        detail=f"unknown reranker kind '{spec.kind}'",
    )


def update_status_from_check(
    spec: Union[TokenizerSpec, ProviderSpec, RerankerSpec],
    check: CheckResult,
) -> None:
    """Mutate `spec.status`, `spec.detail`, `spec.last_checked_at` in place."""
    if check.ok:
        spec.status = Status.STABLE
        spec.detail = None
    else:
        spec.status = (
            Status.UNREACHABLE
            if "unreachable" in check.detail.lower()
            else Status.UNAVAILABLE
        )
        spec.detail = check.detail
    spec.last_checked_at = now_iso()


def validate_template_picks(
    template: Template,
    tokenizer_id: str,
    provider_id: str,
    reranker_id: Optional[str] = None,
) -> ValidationReport:
    """
    Run the lazy-validation flow used by `EmbedderFactory.from_template`.

    Side effect: mutates the `status` / `detail` / `last_checked_at`
    fields of the matched specs on `template` so the caller can persist
    the updated registry.
    """
    report = ValidationReport()

    tok = template.tokenizers.get(tokenizer_id)
    if tok is None:
        report.add(
            f"tokenizer:{tokenizer_id}",
            False,
            f"not present in template '{template.model_key}'",
        )
    else:
        check = validate_tokenizer(tok)
        update_status_from_check(tok, check)
        report.checks.append(check)

    prov = template.providers.get(provider_id)
    if prov is None:
        report.add(
            f"provider:{provider_id}",
            False,
            f"not present in template '{template.model_key}'",
        )
    else:
        check = validate_provider(prov)
        update_status_from_check(prov, check)
        report.checks.append(check)

    if reranker_id is not None:
        rr = template.rerankers.get(reranker_id)
        if rr is None:
            report.add(
                f"reranker:{reranker_id}",
                False,
                f"not present in template '{template.model_key}'",
            )
        else:
            check = validate_reranker(rr)
            update_status_from_check(rr, check)
            report.checks.append(check)

    return report


def validate_full_template(template: Template) -> ValidationReport:
    """Run every tokenizer + provider + reranker in the template; mutates statuses."""
    report = ValidationReport()
    for tok in template.tokenizers.values():
        check = validate_tokenizer(tok)
        update_status_from_check(tok, check)
        report.checks.append(check)
    for prov in template.providers.values():
        check = validate_provider(prov)
        update_status_from_check(prov, check)
        report.checks.append(check)
    for rr in template.rerankers.values():
        check = validate_reranker(rr)
        update_status_from_check(rr, check)
        report.checks.append(check)
    return report
