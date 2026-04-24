"""
Kaggle notebook generation utilities.

Creates competition notebooks (.ipynb) and kernel metadata bundles
for submission via the Kaggle API.
"""

import json
import os
from pathlib import Path
from typing import Any

import nbformat


def create_competition_notebook(
    competition: str,
    code_cells: list[str],
    title: str | None = None,
    markdown_header: str | None = None,
) -> nbformat.NotebookNode:
    """Create a Jupyter notebook for a Kaggle competition.

    Args:
        competition: Kaggle competition slug (e.g. "titanic").
        code_cells: List of Python code strings, one per cell.
        title: Optional notebook title (defaults to competition slug).
        markdown_header: Optional markdown cell at the top.

    Returns:
        nbformat.NotebookNode ready for writing.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    title = title or f"{competition} — agent submission"

    # Header cell
    header_md = markdown_header or f"# {title}\n\nAuto-generated for competition: `{competition}`"
    nb.cells.append(nbformat.v4.new_markdown_cell(header_md))

    for code in code_cells:
        nb.cells.append(nbformat.v4.new_code_cell(code))

    return nb


def write_kernel_bundle(
    notebook: nbformat.NotebookNode,
    competition: str,
    dest_dir: str | Path,
    username: str | None = None,
    title: str | None = None,
    gpu: bool = False,
    internet: bool = True,
    dataset_sources: list[str] | None = None,
    competition_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Write a Kaggle kernel bundle (notebook + kernel-metadata.json).

    Args:
        notebook: The notebook node to write.
        competition: Competition slug.
        dest_dir: Directory to write files into (created if needed).
        username: Kaggle username (falls back to KAGGLE_USERNAME env var).
        title: Kernel title (defaults to competition slug).
        gpu: Whether to request GPU accelerator.
        internet: Whether to enable internet access.
        dataset_sources: Optional list of dataset sources (e.g. ["owner/dataset"]).
        competition_sources: Optional list of competition data sources.

    Returns:
        Dict with keys "notebook_path", "metadata_path", "kernel_ref" for the
        caller to use when pushing via the Kaggle API.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    username = username or os.environ.get("KAGGLE_USERNAME", "unknown")
    title = title or f"{competition}-agent"
    slug = title.lower().replace(" ", "-")[:50]
    kernel_ref = f"{username}/{slug}"

    # Write notebook
    nb_path = dest / "notebook.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    # Write kernel-metadata.json
    metadata: dict[str, Any] = {
        "id": kernel_ref,
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": competition_sources or [competition],
        "kernel_sources": [],
    }
    meta_path = dest / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "notebook_path": str(nb_path),
        "metadata_path": str(meta_path),
        "kernel_ref": kernel_ref,
    }
