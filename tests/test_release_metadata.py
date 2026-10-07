from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

from ragops_lab import __version__

ROOT = Path(__file__).resolve().parents[1]
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21805398"
RELEASE_TITLE = (
    "RAG Quality Engine: An Evaluation-First Platform for Retrieval-Augmented Generation Quality"
)


def test_release_metadata_versions_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation: dict[str, Any] = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    version = pyproject["project"]["version"]
    latest_changelog_match = re.search(
        r"^## \[(?P<version>\d+\.\d+\.\d+)\]", changelog, re.MULTILINE
    )

    assert version == __version__
    assert version == zenodo["version"]
    assert version == citation["version"]
    assert latest_changelog_match is not None
    assert version == latest_changelog_match.group("version")


def test_release_metadata_title_and_description_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation: dict[str, Any] = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert zenodo["title"] == RELEASE_TITLE
    assert citation["title"] == RELEASE_TITLE
    assert pyproject["project"]["description"] in zenodo["description"]
    assert zenodo["description"] == citation["abstract"]


def test_release_metadata_uses_zenodo_concept_doi() -> None:
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation: dict[str, Any] = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    related_identifiers = {
        identifier["identifier"]
        for identifier in zenodo["related_identifiers"]
        if identifier["relation"] == "isVersionOf"
    }

    assert citation["doi"] == ZENODO_CONCEPT_DOI
    assert ZENODO_CONCEPT_DOI in related_identifiers
    assert ZENODO_CONCEPT_DOI in readme
