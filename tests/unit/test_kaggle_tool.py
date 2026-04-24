"""Tests for the Kaggle tool — auth, score persistence, handler dispatch, notebook generation."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.kaggle_tool import (
    KAGGLE_TOOL_SPEC,
    _kaggle_auth_header,
    _load_scores,
    _require_auth,
    _save_score,
    _SCORES_DIR,
    kaggle_handler,
)
from agent.tools.kaggle_notebooks import (
    create_competition_notebook,
    write_kernel_bundle,
)


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_auth_header_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setenv("KAGGLE_KEY", "abc123")
        headers = _kaggle_auth_header()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_auth_header_missing_env_vars(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        headers = _kaggle_auth_header()
        assert headers == {}

    def test_require_auth_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        with pytest.raises(ValueError, match="Kaggle credentials not found"):
            _require_auth()

    def test_require_auth_returns_header_when_set(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "user")
        monkeypatch.setenv("KAGGLE_KEY", "key")
        headers = _require_auth()
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# Score persistence tests
# ---------------------------------------------------------------------------


class TestScorePersistence:
    def test_save_and_load_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.kaggle_tool._SCORES_DIR", tmp_path)
        assert _load_scores("test-comp") == []

        entry = {"score": "0.85", "hypothesis": "baseline"}
        _save_score("test-comp", entry)
        scores = _load_scores("test-comp")
        assert len(scores) == 1
        assert scores[0]["score"] == "0.85"

    def test_multiple_saves_accumulate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.kaggle_tool._SCORES_DIR", tmp_path)
        _save_score("comp2", {"score": "0.9"})
        _save_score("comp2", {"score": "0.85"})
        _save_score("comp2", {"score": "0.80"})
        scores = _load_scores("comp2")
        assert len(scores) == 3

    def test_separate_competitions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.kaggle_tool._SCORES_DIR", tmp_path)
        _save_score("comp-a", {"score": "0.9"})
        _save_score("comp-b", {"score": "0.5"})
        assert len(_load_scores("comp-a")) == 1
        assert len(_load_scores("comp-b")) == 1


# ---------------------------------------------------------------------------
# Handler dispatch tests
# ---------------------------------------------------------------------------


class TestHandlerDispatch:
    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self):
        output, success = await kaggle_handler({"operation": "nonexistent"})
        assert not success
        assert "Unknown operation" in output

    @pytest.mark.asyncio
    async def test_missing_operation_returns_error(self):
        output, success = await kaggle_handler({})
        assert not success
        assert "Unknown operation" in output

    @pytest.mark.asyncio
    async def test_score_history_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.kaggle_tool._SCORES_DIR", tmp_path)
        output, success = await kaggle_handler({
            "operation": "score_history",
            "competition": "some-comp",
        })
        assert success  # not an error, just empty
        assert "No local score history" in output

    @pytest.mark.asyncio
    async def test_score_history_with_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.kaggle_tool._SCORES_DIR", tmp_path)
        _save_score("tracked", {"score": "0.90", "hypothesis": "baseline", "timestamp": "2025-01-01T00:00:00"})
        _save_score("tracked", {"score": "0.85", "hypothesis": "feature eng", "timestamp": "2025-01-02T00:00:00"})
        output, success = await kaggle_handler({
            "operation": "score_history",
            "competition": "tracked",
        })
        assert success
        assert "Best" in output
        assert "0.85" in output
        assert "improving" in output

    @pytest.mark.asyncio
    async def test_submit_missing_competition(self):
        output, success = await kaggle_handler({
            "operation": "submit",
            "file_path": "/tmp/sub.csv",
        })
        assert not success
        assert "competition" in output.lower()

    @pytest.mark.asyncio
    async def test_submit_missing_file(self):
        output, success = await kaggle_handler({
            "operation": "submit",
            "competition": "test",
        })
        assert not success
        assert "file_path" in output.lower()

    @pytest.mark.asyncio
    async def test_submit_file_not_found(self):
        output, success = await kaggle_handler({
            "operation": "submit",
            "competition": "test",
            "file_path": "/nonexistent/path/sub.csv",
        })
        assert not success
        assert "not found" in output.lower()

    @pytest.mark.asyncio
    async def test_list_competitions_requires_auth(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        output, success = await kaggle_handler({"operation": "list_competitions"})
        assert not success
        assert "credentials" in output.lower() or "kaggle" in output.lower()


# ---------------------------------------------------------------------------
# Tool spec tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_has_required_fields(self):
        assert KAGGLE_TOOL_SPEC["name"] == "kaggle"
        assert "description" in KAGGLE_TOOL_SPEC
        assert "parameters" in KAGGLE_TOOL_SPEC

    def test_all_operations_in_enum(self):
        enum_ops = KAGGLE_TOOL_SPEC["parameters"]["properties"]["operation"]["enum"]
        assert "list_competitions" in enum_ops
        assert "submit" in enum_ops
        assert "score_history" in enum_ops
        assert "read_notebook" in enum_ops
        assert len(enum_ops) == 11

    def test_operation_is_required(self):
        assert "operation" in KAGGLE_TOOL_SPEC["parameters"]["required"]


# ---------------------------------------------------------------------------
# Notebook generation tests
# ---------------------------------------------------------------------------


class TestNotebookGeneration:
    def test_create_competition_notebook(self):
        nb = create_competition_notebook(
            competition="titanic",
            code_cells=["import pandas as pd", "df = pd.read_csv('train.csv')"],
        )
        assert len(nb.cells) == 3  # 1 markdown header + 2 code cells
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"
        assert "pandas" in nb.cells[1].source

    def test_create_notebook_custom_title(self):
        nb = create_competition_notebook(
            competition="titanic",
            code_cells=["print('hello')"],
            title="My Custom Title",
        )
        assert "My Custom Title" in nb.cells[0].source

    def test_write_kernel_bundle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        nb = create_competition_notebook(
            competition="titanic",
            code_cells=["print(1)"],
        )
        result = write_kernel_bundle(
            notebook=nb,
            competition="titanic",
            dest_dir=tmp_path / "bundle",
        )
        assert "notebook_path" in result
        assert "metadata_path" in result
        assert "kernel_ref" in result
        assert result["kernel_ref"].startswith("testuser/")

        # Verify files were written
        nb_path = Path(result["notebook_path"])
        meta_path = Path(result["metadata_path"])
        assert nb_path.exists()
        assert meta_path.exists()

        # Verify metadata content
        meta = json.loads(meta_path.read_text())
        assert meta["competition_sources"] == ["titanic"]
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"

    def test_write_kernel_bundle_gpu(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        nb = create_competition_notebook("comp", ["x = 1"])
        result = write_kernel_bundle(
            notebook=nb,
            competition="comp",
            dest_dir=tmp_path / "gpu_bundle",
            gpu=True,
        )
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["enable_gpu"] is True


# ---------------------------------------------------------------------------
# Approval gate test
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_submit_needs_approval(self):
        from agent.core.agent_loop import _needs_approval

        assert _needs_approval("kaggle", {"operation": "submit"}) is True

    def test_read_operations_no_approval(self):
        from agent.core.agent_loop import _needs_approval

        for op in ["list_competitions", "list_notebooks", "read_notebook",
                    "leaderboard", "competition_details", "score_history"]:
            assert _needs_approval("kaggle", {"operation": op}) is False


# ---------------------------------------------------------------------------
# Research tool integration test
# ---------------------------------------------------------------------------


class TestResearchIntegration:
    def test_kaggle_in_research_tool_names(self):
        from agent.tools.research_tool import RESEARCH_TOOL_NAMES
        assert "kaggle" in RESEARCH_TOOL_NAMES
