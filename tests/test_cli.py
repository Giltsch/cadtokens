"""Tests for the CLI entry point."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_STEP = str(TESTS_DIR / "sample.step")
CLI_MODULE = "cadtokens.cli"


def run_cli(*args) -> subprocess.CompletedProcess:
    """Helper to run the CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        capture_output=True,
        text=True,
        cwd=str(TESTS_DIR.parent),  # run from the package root
        timeout=30,
    )


class TestCLIParse:
    def test_parse_basic(self):
        result = run_cli("parse", SAMPLE_STEP)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["entities"] == 628
        assert data["faces"] == 11
        assert data["edges"] == 21

    def test_parse_full(self):
        result = run_cli("parse", "--full", SAMPLE_STEP)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "face_details" in data
        assert len(data["face_details"]) == 11
        assert data["face_details"][0]["surface_type"] == "PLANE"


class TestCLIPMI:
    def test_pmi_basic(self):
        result = run_cli("pmi", SAMPLE_STEP)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "tolerances" in data
        assert "dimensions" in data
        assert "datums" in data
        assert "summary" in data


class TestCLIHelp:
    def test_no_args_shows_help(self):
        result = run_cli()
        # Should exit non-zero and show help text
        assert result.returncode != 0

    def test_help_flag(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "cadtokens" in result.stdout.lower()

    def test_parse_help(self):
        result = run_cli("parse", "--help")
        assert result.returncode == 0
        assert "step_file" in result.stdout.lower() or "STEP" in result.stdout
