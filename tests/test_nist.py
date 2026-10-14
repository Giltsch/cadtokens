"""Regression tests against the NIST MBE PMI test models (AP242).

These files are the reference set for semantic PMI. Fetch them with
`python scripts/fetch_nist.py`; the tests are skipped if they are missing.
Expected values for FTC-10 were checked against the NIST drawing
(nist_ftc_10_asme1_rb.pdf).
"""

from pathlib import Path

import pytest

import cadtokens

NIST_DIR = Path(__file__).resolve().parent / "data" / "nist"

pytestmark = pytest.mark.skipif(
    not NIST_DIR.is_dir() or not any(NIST_DIR.glob("*.stp")),
    reason="NIST test models missing — run: python scripts/fetch_nist.py",
)

# file: (tolerances, dimensions, datums, datum_systems, length_unit)
EXPECTED_COUNTS = {
    "nist_ctc_01_asme1_ap242-e1.stp": (6, 12, 3, 2, 'mm'),
    "nist_ctc_02_asme1_ap242-e2.stp": (22, 7, 10, 14, 'mm'),
    "nist_ctc_03_asme1_ap242-e2.stp": (13, 10, 6, 6, 'in'),
    "nist_ctc_04_asme1_ap242-e1.stp": (7, 10, 8, 5, 'mm'),
    "nist_ctc_05_asme1_ap242-e1.stp": (10, 6, 4, 5, 'in'),
    "nist_ftc_06_asme1_ap242-e2.stp": (28, 24, 10, 12, 'in'),
    "nist_ftc_07_asme1_ap242-e2.stp": (27, 17, 5, 5, 'in'),
    "nist_ftc_08_asme1_ap242-e1-tg.stp": (0, 0, 0, 0, None),
    "nist_ftc_08_asme1_ap242-e2.stp": (33, 10, 11, 9, 'in'),
    "nist_ftc_09_asme1_ap242-e1.stp": (31, 22, 8, 8, 'mm'),
    "nist_ftc_10_asme1_ap242-e2.stp": (40, 26, 11, 12, 'mm'),
    "nist_ftc_11_asme1_ap242-e2.stp": (4, 6, 2, 1, 'mm'),
    "nist_stc_06_asme1_ap242-e3.stp": (25, 17, 6, 9, 'in'),
    "nist_stc_07_asme1_ap242-e3.stp": (22, 45, 5, 5, 'in'),
    "nist_stc_08_asme1_ap242-e3.stp": (30, 7, 8, 9, 'in'),
    "nist_stc_09_asme1_ap242-e3.stp": (27, 22, 8, 8, 'in'),
    "nist_stc_10_asme1_ap242-e2.stp": (37, 27, 9, 12, 'mm'),
}

_cache = {}


def pmi_of(name):
    if name not in _cache:
        _cache[name] = cadtokens.extract_pmi(cadtokens.load(str(NIST_DIR / name)))
    return _cache[name]


def find_tol(pmi, name):
    matches = [t for t in pmi.tolerances if t.name == name]
    assert matches, f"tolerance {name!r} not found"
    return matches[0]


@pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
def test_counts_and_units(name):
    pmi = pmi_of(name)
    got = (len(pmi.tolerances), len(pmi.dimensions), len(pmi.datums),
           len(pmi.datum_systems), pmi.length_unit)
    assert got == EXPECTED_COUNTS[name]


def test_every_tolerance_has_a_value():
    for name in EXPECTED_COUNTS:
        assert all(t.value is not None for t in pmi_of(name).tolerances), name


def test_geometry_resolution_rate():
    tols = [t for n in EXPECTED_COUNTS for t in pmi_of(n).tolerances]
    dims = [d for n in EXPECTED_COUNTS for d in pmi_of(n).dimensions]
    linked = [t for t in tols if t.face_ids or t.edge_ids or t.other_geometry_ids or t.whole_part]
    # The two unlinked tolerances have no geometry association in the file itself
    assert len(linked) >= len(tols) - 2
    assert all(d.face_ids or d.edge_ids or d.other_geometry_ids for d in dims)


class TestFTC10AgainstDrawing:
    """Feature control frames as printed on nist_ftc_10_asme1_rb.pdf."""
    name = "nist_ftc_10_asme1_ap242-e2.stp"

    def test_position_diametral_two_datums(self):
        t = find_tol(pmi_of(self.name), "Position.1")      # ⌖ ⌀0.5 | A | B
        assert (t.tolerance_type, t.value, t.unit) == ("POSITION", 0.5, "mm")
        assert t.is_diametral and t.datum_labels == ["A", "B"]

    def test_pattern_count(self):
        t = find_tol(pmi_of(self.name), "Position.3")      # 8X ⌀2.5 ⌖ ⌀0.75 | A | B | C
        assert t.value == 0.75 and t.feature_count == 8
        assert t.datum_labels == ["A", "B", "C"]

    def test_profile_all_over_applies_to_whole_part(self):
        t = find_tol(pmi_of(self.name), "Geometrical Tolerance.1")   # note 5: profile 1.5 A|B|C
        assert t.tolerance_type == "SURFACE_PROFILE" and t.value == 1.5 and t.whole_part

    def test_simple_form_keeps_datum(self):
        t = find_tol(pmi_of(self.name), "Perpendicularity.1")   # ⟂ ⌀0.25 | A
        assert not t.is_complex and t.datum_labels == ["A"] and t.is_diametral

    def test_common_datum_with_mmb(self):
        t = find_tol(pmi_of(self.name), "Position.10")     # ⌖ ⌀0.8 | D | E(M)-F(M)-G(M)
        assert [r.label for r in t.datum_references] == ["D", "E-F-G"]
        assert t.datum_references[1].modifiers == ["MAXIMUM_MATERIAL_REQUIREMENT"]

    def test_zero_at_mmc_and_lmc(self):
        pmi = pmi_of(self.name)
        assert find_tol(pmi, "Position.11").modifiers == ["MAXIMUM_MATERIAL_REQUIREMENT"]
        t = find_tol(pmi, "Position.12")                   # 3X ⌖ ⌀0Ⓛ | H | K | L
        assert t.value == 0.0 and t.modifiers == ["LEAST_MATERIAL_REQUIREMENT"]
        assert t.feature_count == 3 and t.datum_labels == ["H", "K", "L"]

    def test_projected_zone(self):
        t = find_tol(pmi_of(self.name), "Position.16")     # 2X M3 ⌖ ⌀0.8 Ⓟ3.2 | D | E | B
        assert t.projected_length == 3.2 and t.feature_count == 2
        assert t.datum_labels == ["D", "E", "B"]

    def test_straightness_per_unit_length(self):
        t = find_tol(pmi_of(self.name), "Straightness.1")  # — 0.2/15
        assert t.value == 0.2 and t.unit_basis == [15.0]

    def test_fit_dimension(self):
        fits = [d for d in pmi_of(self.name).dimensions if d.fit]
        assert len(fits) == 3                               # ⌀3.5 G6 (3 holes)
        assert all(d.nominal_value == 3.5 and d.fit == "G6" and d.fit_kind == "hole" for d in fits)

    def test_plus_minus_dimension(self):
        d = [d for d in pmi_of(self.name).dimensions if d.feature_count == 8][0]   # 8X ⌀2.5 ±0.15
        assert (d.nominal_value, d.lower_tolerance, d.upper_tolerance) == (2.5, -0.15, 0.15)
        assert (d.min_value, d.max_value) == pytest.approx((2.35, 2.65))


class TestOtherSemantics:
    def test_limit_dimensions(self):
        dims = pmi_of("nist_stc_07_asme1_ap242-e3.stp").dimensions
        limits = [d for d in dims if d.upper_limit is not None]
        assert len(limits) == 32
        assert all(d.lower_limit < d.upper_limit for d in limits)

    def test_max_qualifier(self):
        dims = pmi_of("nist_ftc_11_asme1_ap242-e2.stp").dimensions
        assert any(d.semantic_name == "radius" and d.qualifier == "maximum" for d in dims)

    def test_area_unit_basis(self):
        t = find_tol(pmi_of("nist_ctc_03_asme1_ap242-e2.stp"), "Flatness.1")
        assert t.unit_basis == [0.25, 0.25] and t.unit_basis_area_type == "RECTANGULAR"

    def test_unequally_disposed(self):
        pmi = pmi_of("nist_ctc_02_asme1_ap242-e2.stp")
        assert sorted(t.unequal_displacement for t in pmi.tolerances
                      if t.unequal_displacement is not None) == [0.0, 0.5, 2.5]

    def test_mixed_unit_export_is_flagged(self):
        # FTC-9 is an inch drawing; the file links tolerance values to mm
        warnings = pmi_of("nist_ftc_09_asme1_ap242-e1.stp").warnings
        assert any("Mixed length units" in w for w in warnings)

    def test_unlinked_annotation_is_flagged(self):
        warnings = pmi_of("nist_ftc_06_asme1_ap242-e2.stp").warnings
        assert any("not linked to any geometry" in w for w in warnings)
