"""Shared project configuration models."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

RETRIEVAL_MODES = frozenset({"lexical", "vector", "hybrid"})
RETRIEVAL_FUSION_STRATEGIES = frozenset({"weighted", "rrf"})
LLM_PROVIDERS = frozenset({"heuristic", "openai-compatible"})
EMBEDDING_PROVIDERS = frozenset({"fake", "sentence-transformers"})


class ProjectPaths(BaseModel):
    """Filesystem paths used by the project.

    The model keeps path validation explicit so CLI and services fail early when
    a required directory is missing or incorrectly configured.
    """

    data_dir: Path = Field(default=Path("data"), description="Base data directory.")
    artifact_dir: Path = Field(default=Path("artifacts"), description="Generated artifacts.")
    model_dir: Path = Field(default=Path("models"), description="Trained model storage.")
    trace_path: Path = Field(
        default=Path("artifacts/traces/traces.jsonl"),
        description="Trace storage path.",
    )
    chunk_path: Path = Field(
        default=Path("data/processed/chunks.jsonl"),
        description="Default chunk storage path.",
    )
    vector_index_path: Path = Field(
        default=Path("artifacts/index/vector_index.json"),
        description="Default local vector index path.",
    )

    @field_validator(
        "data_dir",
        "artifact_dir",
        "model_dir",
        "trace_path",
        "chunk_path",
        "vector_index_path",
    )
    @classmethod
    def ensure_relative_or_absolute_path(cls, value: Path) -> Path:
        """Validate path-like fields.

        Parameters
        ----------
        value:
            Candidate path.

        Returns
        -------
        Path
            The validated path.
        """
        if not isinstance(value, Path):
            raise TypeError("Expected pathlib.Path instance.")
        return value


class LLMSettings(BaseModel):
    """LLM provider settings for generation clients."""

    provider: str = Field(default="heuristic")
    model: str = Field(default="heuristic-grounded", min_length=1)
    endpoint: str | None = Field(default=None)
    api_key_env: str = Field(default="OPENAI_API_KEY", min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in LLM_PROVIDERS:
            supported = ", ".join(sorted(LLM_PROVIDERS))
            raise ValueError(f"Unsupported LLM provider: {value}. Supported: {supported}.")
        return normalized


class EmbeddingSettings(BaseModel):
    """Embedding provider settings for vector indexing and search."""

    provider: str = Field(default="fake")
    model: str = Field(default="fake-bow", min_length=1)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in EMBEDDING_PROVIDERS:
            supported = ", ".join(sorted(EMBEDDING_PROVIDERS))
            raise ValueError(f"Unsupported embedding provider: {value}. Supported: {supported}.")
        return normalized


class RetrievalProfile(BaseModel):
    """Retrieval configuration decoupled from runtime paths."""

    name: str = Field(min_length=1)
    mode: str = Field(default="lexical")
    top_k: int = Field(default=5, ge=1)
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion_strategy: str = Field(default="weighted")
    rrf_k: int = Field(default=60, ge=1)
    rerank: bool = Field(default=False)
    rerank_candidate_multiplier: int = Field(default=4, ge=1)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in RETRIEVAL_MODES:
            supported = ", ".join(sorted(RETRIEVAL_MODES))
            raise ValueError(f"Unsupported retrieval mode: {value}. Supported: {supported}.")
        return normalized

    @field_validator("fusion_strategy")
    @classmethod
    def validate_fusion_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in RETRIEVAL_FUSION_STRATEGIES:
            supported = ", ".join(sorted(RETRIEVAL_FUSION_STRATEGIES))
            raise ValueError(
                f"Unsupported fusion strategy: {value}. Supported: {supported}."
            )
        return normalized


class RuntimeSettings(BaseModel):
    """Runtime settings shared by notebooks, CLI, and API components."""

    environment: str = Field(default="local")
    random_seed: int = Field(default=42, ge=0)
    debug: bool = Field(default=False)
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    api_max_request_bytes: int = Field(default=1_000_000, gt=0)
    api_max_top_k: int = Field(default=20, ge=1)
    api_max_query_chars: int = Field(default=1_000, ge=1)
    api_max_text_chars: int = Field(default=20_000, ge=1)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval_profiles: dict[str, RetrievalProfile] = Field(
        default_factory=lambda: {
            "lexical": RetrievalProfile(name="lexical", mode="lexical", top_k=5),
            "vector": RetrievalProfile(name="vector", mode="vector", top_k=5),
            "hybrid": RetrievalProfile(
                name="hybrid",
                mode="hybrid",
                top_k=5,
                lexical_weight=0.5,
                vector_weight=0.5,
            ),
        }
    )

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        """Build runtime settings from `RAGOPS_*` environment variables."""
        paths = ProjectPaths(
            data_dir=_path_from_env(
                "RAGOPS_DATA_DIR",
                ProjectPaths.model_fields["data_dir"].default,
            ),
            artifact_dir=_path_from_env(
                "RAGOPS_ARTIFACT_DIR",
                ProjectPaths.model_fields["artifact_dir"].default,
            ),
            model_dir=_path_from_env(
                "RAGOPS_MODEL_DIR",
                ProjectPaths.model_fields["model_dir"].default,
            ),
            trace_path=_path_from_env(
                "RAGOPS_TRACE_PATH",
                ProjectPaths.model_fields["trace_path"].default,
            ),
            chunk_path=_path_from_env(
                "RAGOPS_CHUNK_PATH",
                ProjectPaths.model_fields["chunk_path"].default,
            ),
            vector_index_path=_path_from_env(
                "RAGOPS_VECTOR_INDEX_PATH",
                ProjectPaths.model_fields["vector_index_path"].default,
            ),
        )
        return cls(
            environment=os.getenv("RAGOPS_ENVIRONMENT", "local"),
            random_seed=_int_from_env("RAGOPS_RANDOM_SEED", 42),
            debug=_bool_from_env("RAGOPS_DEBUG", default=False),
            paths=paths,
            api_max_request_bytes=_int_from_env("RAGOPS_API_MAX_REQUEST_BYTES", 1_000_000),
            api_max_top_k=_int_from_env("RAGOPS_API_MAX_TOP_K", 20),
            api_max_query_chars=_int_from_env("RAGOPS_API_MAX_QUERY_CHARS", 1_000),
            api_max_text_chars=_int_from_env("RAGOPS_API_MAX_TEXT_CHARS", 20_000),
            llm=LLMSettings(
                provider=os.getenv("RAGOPS_LLM_PROVIDER", "heuristic"),
                model=os.getenv("RAGOPS_LLM_MODEL", "heuristic-grounded"),
                endpoint=os.getenv("RAGOPS_LLM_ENDPOINT"),
                api_key_env=os.getenv("RAGOPS_LLM_API_KEY_ENV", "OPENAI_API_KEY"),
                timeout_seconds=_float_from_env("RAGOPS_LLM_TIMEOUT_SECONDS", 30.0),
            ),
            embeddings=EmbeddingSettings(
                provider=os.getenv("RAGOPS_EMBEDDING_PROVIDER", "fake"),
                model=os.getenv("RAGOPS_EMBEDDING_MODEL", "fake-bow"),
            ),
        )

    def resolve_retrieval_profile(
        self,
        name: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        lexical_weight: float | None = None,
        vector_weight: float | None = None,
        fusion_strategy: str | None = None,
        rrf_k: int | None = None,
        rerank: bool | None = None,
        rerank_candidate_multiplier: int | None = None,
    ) -> RetrievalProfile:
        """Resolve a named profile with optional runtime overrides."""
        profile = self.retrieval_profiles.get(name)
        if profile is None:
            available = ", ".join(sorted(self.retrieval_profiles))
            raise ValueError(f"Unknown retrieval profile: {name}. Available: {available}.")
        overrides = {
            key: value
            for key, value in {
                "mode": mode,
                "top_k": top_k,
                "lexical_weight": lexical_weight,
                "vector_weight": vector_weight,
                "fusion_strategy": fusion_strategy,
                "rrf_k": rrf_k,
                "rerank": rerank,
                "rerank_candidate_multiplier": rerank_candidate_multiplier,
            }.items()
            if value is not None
        }
        return RetrievalProfile.model_validate(
            profile.model_dump() | overrides,
        )


def _path_from_env(name: str, default: object) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value)
    if isinstance(default, Path):
        return default
    raise TypeError(f"Default for {name} must be a pathlib.Path instance.")


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _bool_from_env(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")
