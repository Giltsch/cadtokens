"""conftest.py — shared fixtures for cadtokens tests."""

import os
from pathlib import Path

import pytest

# Resolve the tests/ directory once
TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_STEP = TESTS_DIR / "sample.step"


@pytest.fixture
def sample_step_path():
    """Path to the sample STEP file shipped with the test suite."""
    assert SAMPLE_STEP.exists(), f"Missing test fixture: {SAMPLE_STEP}"
    return str(SAMPLE_STEP)


@pytest.fixture
def step_file(sample_step_path):
    """Parsed StepFile from the sample."""
    import cadtokens
    return cadtokens.load(sample_step_path)


@pytest.fixture
def ref_graph(step_file):
    """ReferenceGraph built from the sample."""
    import cadtokens
    return cadtokens.graph(step_file)


@pytest.fixture
def brep(ref_graph):
    """BRepExtractor for the sample."""
    import cadtokens
    return cadtokens.BRepExtractor(ref_graph)
