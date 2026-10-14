"""Tests for the ballooned PDF report."""

import math
import re

import pytest

import cadtokens
from cadtokens.report import (
    Canvas, _ring_angles, _tokens, rich_width, write_pdf, write_report,
)
from cadtokens.step_engine.wireframe import Wireframe, _de_boor
from tests.test_nist import NIST_DIR, pytestmark as nist_mark


def _pdf_pages(path):
    data = path.read_bytes()
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    xref = int(re.search(rb"startxref\s+(\d+)", data).group(1))
    assert data[xref:xref + 4] == b"xref"
    return len(re.findall(rb"/Type /Page\b", data))


def _min_gap(angles):
    s = sorted(a % (2 * math.pi) for a in angles)
    return min((s[(i + 1) % len(s)] - s[i]) % (2 * math.pi) for i in range(len(s)))


@pytest.mark.parametrize("ideal", [
    [0.1, 0.1, 0.1, 0.12, 3.0, 3.1, -3.1],
    [0.0] * 5,
    [3.1, -3.1, 3.12, -3.12],
    [i * 0.05 for i in range(20)],
])
def test_ring_angles_spacing_and_locality(ideal):
    angles = _ring_angles(ideal, 0.2)
    assert _min_gap(angles) >= 0.2 - 1e-9
    shift = max(abs(((a - i + math.pi) % (2 * math.pi)) - math.pi) for a, i in zip(angles, ideal))
    assert shift <= 0.2 * len(ideal) / 2 + 1e-9


def test_ring_angles_too_many_spreads_evenly():
    angles = _ring_angles([1.0] * 40, 0.2)
    assert _min_gap(angles) == pytest.approx(2 * math.pi / 40)


def test_spec_tokens():
    toks = _tokens("4X ⌖ ⌀0.5 Ⓜ Ⓟ3.2")
    kinds = [t[0] for t in toks]
    assert kinds == ["t", "gdt", "t", "diam", "t", "circ", "t", "circ", "t"]
    assert rich_width("⌀6 ±0.15", 7) > 0


def test_pdf_writer_escapes_text(tmp_path):
    c = Canvas()
    c.text(10, 10, "a (b) \\ ±0.1 °")
    out = tmp_path / "t.pdf"
    write_pdf([c], str(out), title="x (y)")
    assert _pdf_pages(out) == 1


def test_bspline_evaluation_hits_endpoints():
    cps = [(0, 0, 0), (1, 2, 0), (3, 2, 0), (4, 0, 0)]
    knots = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _de_boor(3, knots, cps, None, 0.0) == pytest.approx((0, 0, 0))
    assert _de_boor(3, knots, cps, None, 1.0) == pytest.approx((4, 0, 0))
    mid = _de_boor(3, knots, cps, None, 0.5)
    assert mid == pytest.approx((2.0, 1.5, 0.0))


def test_report_without_pmi(sample_step_path, tmp_path):
    out = tmp_path / "sample.pdf"
    info = write_report(sample_step_path, str(out))
    assert info["characteristics"] == 0 and _pdf_pages(out) == info["pages"] == 2


@nist_mark
def test_nist_report(tmp_path):
    out = tmp_path / "ftc10.pdf"
    info = write_report(str(NIST_DIR / "nist_ftc_10_asme1_ap242-e2.stp"), str(out))
    assert info["characteristics"] == 66
    assert info["ballooned"] == 59          # 7 are whole-part / supplemental geometry
    assert _pdf_pages(out) == info["pages"] == 4


@nist_mark
def test_wireframe_samples_every_edge_with_its_curve():
    sf = cadtokens.load(str(NIST_DIR / "nist_ftc_10_asme1_ap242-e2.stp"))
    g = cadtokens.graph(sf)
    edges = Wireframe(g).edges()
    assert len(edges) == len(g.find_by_type("EDGE_CURVE"))
    circles = [e for e in edges if sf[sf[e].params[3].entity_id].has_type("SURFACE_CURVE")
               or sf[sf[e].params[3].entity_id].has_type("CIRCLE")]
    assert any(len(edges[e]) > 2 for e in circles)
