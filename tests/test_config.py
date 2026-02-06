"""Tests for automem.config helper functions."""

from __future__ import annotations

import pytest

from automem.config import (
    get_openai_token_params,
    normalize_memory_type,
    supports_reasoning,
)


class TestSupportsReasoning:
    """Tests for the supports_reasoning helper."""

    def test_o1_models_support_reasoning(self):
        assert supports_reasoning("o1") is True
        assert supports_reasoning("o1-mini") is True
        assert supports_reasoning("o1-preview") is True

    def test_o3_models_support_reasoning(self):
        assert supports_reasoning("o3") is True
        assert supports_reasoning("o3-mini") is True

    def test_gpt5_models_support_reasoning(self):
        assert supports_reasoning("gpt-5") is True
        assert supports_reasoning("gpt-5.2") is True
        assert supports_reasoning("gpt-5.3-codex") is True

    def test_gpt4_models_do_not_support_reasoning(self):
        assert supports_reasoning("gpt-4") is False
        assert supports_reasoning("gpt-4o") is False
        assert supports_reasoning("gpt-4o-mini") is False
        assert supports_reasoning("gpt-4-turbo") is False

    def test_other_models_do_not_support_reasoning(self):
        assert supports_reasoning("text-embedding-3-large") is False
        assert supports_reasoning("claude-3-opus") is False
        assert supports_reasoning("unknown-model") is False


class TestGetOpenaiTokenParams:
    """Tests for the get_openai_token_params helper."""

    def test_o_series_uses_max_completion_tokens(self):
        params = get_openai_token_params("o1", 100)
        assert params == {"max_completion_tokens": 100}

    def test_o_series_with_reasoning(self):
        params = get_openai_token_params("o1", 100, "medium")
        assert params == {"max_completion_tokens": 100, "reasoning": {"effort": "medium"}}

    def test_gpt4_uses_max_tokens(self):
        params = get_openai_token_params("gpt-4o-mini", 100)
        assert params == {"max_tokens": 100}

    def test_gpt5_uses_max_tokens(self):
        params = get_openai_token_params("gpt-5.2", 100)
        assert params == {"max_tokens": 100}

    def test_gpt5_with_reasoning(self):
        params = get_openai_token_params("gpt-5.2", 100, "high")
        assert params == {"max_tokens": 100, "reasoning": {"effort": "high"}}

    def test_gpt4_ignores_reasoning(self):
        # GPT-4 doesn't support reasoning, so it should be ignored
        params = get_openai_token_params("gpt-4o-mini", 100, "medium")
        assert params == {"max_tokens": 100}
        assert "reasoning" not in params

    def test_none_reasoning_is_ignored(self):
        params = get_openai_token_params("gpt-5.2", 100, None)
        assert params == {"max_tokens": 100}
        assert "reasoning" not in params

    def test_empty_string_reasoning_is_ignored(self):
        params = get_openai_token_params("gpt-5.2", 100, "")
        assert params == {"max_tokens": 100}
        assert "reasoning" not in params

    def test_all_reasoning_levels(self):
        for level in ["low", "medium", "high", "xhigh"]:
            params = get_openai_token_params("o3", 50, level)
            assert params["reasoning"]["effort"] == level


class TestNormalizeMemoryType:
    """Tests for the normalize_memory_type helper."""

    def test_canonical_types_unchanged(self):
        for type_name in ["Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context"]:
            result, modified = normalize_memory_type(type_name)
            assert result == type_name
            assert modified is False

    def test_lowercase_types_normalized(self):
        result, modified = normalize_memory_type("decision")
        assert result == "Decision"
        assert modified is True

    def test_legacy_types_normalized(self):
        result, modified = normalize_memory_type("memory")
        assert result == "Context"
        assert modified is True

        result, modified = normalize_memory_type("analysis")
        assert result == "Insight"
        assert modified is True

    def test_none_defaults_to_context(self):
        result, modified = normalize_memory_type(None)
        assert result == "Context"
        assert modified is True

    def test_unknown_type_returns_empty(self):
        result, modified = normalize_memory_type("unknown_type")
        assert result == ""
        assert modified is True
