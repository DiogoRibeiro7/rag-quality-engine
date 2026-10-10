from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from ragops_lab.config import ProjectPaths, RetrievalProfile, RuntimeSettings


def test_runtime_settings_defaults_are_valid() -> None:
    settings = RuntimeSettings()

    assert settings.environment == "local"
    assert settings.random_seed >= 0
    assert isinstance(settings.paths, ProjectPaths)
    assert settings.retrieval_profiles["hybrid"].mode == "hybrid"


def test_retrieval_profile_rejects_unsupported_modes() -> None:
    try:
        RetrievalProfile(name="bad", mode="rerank")
    except ValueError as exc:
        assert "Unsupported retrieval mode" in str(exc)
    else:
        raise AssertionError("Expected unsupported retrieval mode to fail.")


def test_runtime_settings_resolves_retrieval_profile_overrides() -> None:
    settings = RuntimeSettings()

    profile = settings.resolve_retrieval_profile(
        "hybrid",
        top_k=3,
        lexical_weight=0.8,
        vector_weight=0.2,
        rerank=True,
        reranker_strategy="embedding",
        rerank_candidate_multiplier=3,
    )

    assert profile.mode == "hybrid"
    assert profile.top_k == 3
    assert profile.lexical_weight == 0.8
    assert profile.vector_weight == 0.2
    assert profile.rerank is True
    assert profile.reranker_strategy == "embedding"
    assert profile.rerank_candidate_multiplier == 3


def test_runtime_settings_reports_unknown_retrieval_profiles() -> None:
    settings = RuntimeSettings()

    try:
        settings.resolve_retrieval_profile("missing")
    except ValueError as exc:
        assert "Unknown retrieval profile" in str(exc)
    else:
        raise AssertionError("Expected unknown retrieval profile to fail.")


def test_project_paths_accept_path_objects() -> None:
    paths = ProjectPaths(
        data_dir=Path("data"), artifact_dir=Path("artifacts"), model_dir=Path("models")
    )

    assert paths.data_dir == Path("data")
    assert paths.artifact_dir == Path("artifacts")
    assert paths.model_dir == Path("models")


def test_runtime_settings_can_load_environment_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RAGOPS_ENVIRONMENT", "ci")
    monkeypatch.setenv("RAGOPS_CHUNK_PATH", "custom/chunks.jsonl")
    monkeypatch.setenv("RAGOPS_TRACE_PATH", "custom/traces.jsonl")
    monkeypatch.setenv("RAGOPS_VECTOR_INDEX_PATH", "custom/vector_index.json")
    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("RAGOPS_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("RAGOPS_LLM_ENDPOINT", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("RAGOPS_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("RAGOPS_EMBEDDING_MODEL", "fake-test")
    monkeypatch.setenv("RAGOPS_API_MAX_REQUEST_BYTES", "2048")
    monkeypatch.setenv("RAGOPS_API_MAX_TOP_K", "7")
    monkeypatch.setenv("RAGOPS_API_MAX_QUERY_CHARS", "120")

    settings = RuntimeSettings.from_env()

    assert settings.environment == "ci"
    assert settings.paths.chunk_path == Path("custom/chunks.jsonl")
    assert settings.paths.trace_path == Path("custom/traces.jsonl")
    assert settings.paths.vector_index_path == Path("custom/vector_index.json")
    assert settings.llm.provider == "openai-compatible"
    assert settings.llm.model == "gpt-test"
    assert settings.llm.endpoint == "https://example.test/v1/chat/completions"
    assert settings.embeddings.provider == "fake"
    assert settings.embeddings.model == "fake-test"
    assert settings.api_max_request_bytes == 2048
    assert settings.api_max_top_k == 7
    assert settings.api_max_query_chars == 120


def test_runtime_settings_reject_invalid_environment_integers(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGOPS_API_MAX_TOP_K", "many")

    try:
        RuntimeSettings.from_env()
    except ValueError as exc:
        assert "RAGOPS_API_MAX_TOP_K must be an integer" in str(exc)
    else:
        raise AssertionError("Expected invalid integer environment variable to fail.")


def test_runtime_settings_reject_invalid_environment_booleans(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGOPS_DEBUG", "sometimes")

    try:
        RuntimeSettings.from_env()
    except ValueError as exc:
        assert "RAGOPS_DEBUG must be a boolean value" in str(exc)
    else:
        raise AssertionError("Expected invalid boolean environment variable to fail.")


def test_retrieval_profile_rejects_invalid_reranker_strategy() -> None:
    try:
        RetrievalProfile(name="bad", reranker_strategy="unknown")
    except ValueError as exc:
        assert "Unsupported reranker strategy" in str(exc)
    else:
        raise AssertionError("Expected invalid reranker strategy to fail.")
