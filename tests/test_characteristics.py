"""Tests for the inspection characteristic list."""

import csv
import io
import json

import pytest

import cadtokens
from cadtokens.characteristics import (
    build_characteristics, to_csv, to_json, to_table,
)
from cadtokens.step_engine.pmi import (
    DatumReference, DimensionInfo, PMIData, ToleranceInfo,
)
from tests.test_nist import NIST_DIR, pytestmark as nist_mark


def make_pmi():
    hole = DimensionInfo(entity_id=10, name="DIMENSIONAL_SIZE", dimension_type="DIMENSIONAL_SIZE",
                         semantic_name="diameter", nominal_value=6.0, lower_tolerance=-0.15,
                         upper_tolerance=0.15, face_ids=[101, 102], unit="mm", feature_count=4)
    position = ToleranceInfo(entity_id=20, name="Position.1", tolerance_type="POSITION", value=0.5,
                             face_ids=[101, 102], unit="mm", zone_form="cylindrical or circular",
                             modifiers=["MAXIMUM_MATERIAL_REQUIREMENT"], feature_count=4,
                             datum_references=[DatumReference("A"), DatumReference("B-C", ["MAXIMUM_MATERIAL_REQUIREMENT"])])
    flat = ToleranceInfo(entity_id=5, name="Flatness.1", tolerance_type="FLATNESS", value=0.05,
                         face_ids=[200], unit="mm")
    limit = DimensionInfo(entity_id=30, name="DIMENSIONAL_SIZE", dimension_type="DIMENSIONAL_SIZE",
                          semantic_name="radius", upper_limit=0.022, lower_limit=0.012,
                          face_ids=[300], unit="in", decimals=3)
    unilateral = DimensionInfo(entity_id=40, name="DIMENSIONAL_SIZE", dimension_type="DIMENSIONAL_SIZE",
                               semantic_name="diameter", nominal_value=0.25, lower_tolerance=0.0,
                               upper_tolerance=0.003, face_ids=[400], unit="in", decimals=3)
    fit = DimensionInfo(entity_id=50, name="DIMENSIONAL_SIZE", dimension_type="DIMENSIONAL_SIZE",
                        semantic_name="diameter", nominal_value=3.5, fit="G6", fit_kind="hole",
                        face_ids=[500], unit="mm")
    return PMIData(tolerances=[position, flat], dimensions=[hole, limit, unilateral, fit])


class TestBuild:
    def test_numbering_and_feature_grouping(self):
        chars = build_characteristics(make_pmi())
        assert [c.number for c in chars] == list(range(1, len(chars) + 1))
        # Flatness (entity 5) first; hole size directly followed by its position
        assert chars[0].characteristic == "Flatness"
        assert (chars[1].kind, chars[2].kind) == ("dimension", "geometric_tolerance")
        assert chars[1].face_ids == chars[2].face_ids

    def test_dimension_spec_and_limits(self):
        hole = build_characteristics(make_pmi())[1]
        assert hole.specification == "4X ⌀6 ±0.15"
        assert (hole.lower_limit, hole.upper_limit) == pytest.approx((5.85, 6.15))

    def test_feature_control_frame(self):
        pos = build_characteristics(make_pmi())[2]
        assert pos.specification == "4X ⌖ | ⌀0.5 Ⓜ | A | B-C Ⓜ"
        assert (pos.lower_limit, pos.upper_limit) == (0.0, 0.5)
        assert any("bonus tolerance" in n for n in pos.notes)

    def test_limit_dimension(self):
        r = [c for c in build_characteristics(make_pmi()) if c.source_entity_id == 30][0]
        assert r.specification == "R0.022/0.012"
        assert (r.lower_limit, r.upper_limit, r.unit) == (0.012, 0.022, "in")

    def test_unilateral_tolerance_notation(self):
        r = [c for c in build_characteristics(make_pmi()) if c.source_entity_id == 40][0]
        assert r.specification == "⌀0.250 +0.003/-0.000"

    def test_fit_limits_computed(self):
        r = [c for c in build_characteristics(make_pmi()) if c.source_entity_id == 50][0]
        assert r.specification == "⌀3.5 G6 (+0.012/+0.004)"
        assert (r.lower_limit, r.upper_limit) == pytest.approx((3.504, 3.512))
        assert any("ISO 286-1" in n for n in r.notes)

    def test_fit_in_inch_file_converted(self):
        pmi = make_pmi()
        pmi.dimensions = [DimensionInfo(entity_id=70, name="DIMENSIONAL_SIZE",
                                        dimension_type="DIMENSIONAL_SIZE", semantic_name="diameter",
                                        nominal_value=1.0, fit="H7", face_ids=[1], unit="in")]
        r = [c for c in build_characteristics(pmi) if c.source_entity_id == 70][0]   # 25.4 mm H7: 0/+21 µm
        assert (r.lower_limit, r.upper_limit) == pytest.approx((1.0, 1.0 + 0.021 / 25.4))

    def test_unsupported_fit_is_flagged_not_guessed(self):
        pmi = make_pmi()
        pmi.dimensions = [DimensionInfo(entity_id=80, name="DIMENSIONAL_SIZE",
                                        dimension_type="DIMENSIONAL_SIZE", semantic_name="diameter",
                                        nominal_value=25.0, fit="ZC9", face_ids=[1], unit="mm")]
        r = [c for c in build_characteristics(pmi) if c.source_entity_id == 80][0]
        assert r.specification == "⌀25 ZC9"
        assert r.lower_limit is None and r.upper_limit is None
        assert any("not computed" in n for n in r.notes)

    def test_toleranced_only(self):
        pmi = make_pmi()
        pmi.dimensions.append(DimensionInfo(entity_id=60, name="DIMENSIONAL_LOCATION",
                                            dimension_type="DIMENSIONAL_LOCATION",
                                            nominal_value=25.0, face_ids=[600, 601]))
        assert any(c.source_entity_id == 60 for c in build_characteristics(pmi))
        assert not any(c.source_entity_id == 60
                       for c in build_characteristics(pmi, include_untoleranced=False))


class TestExport:
    def test_csv_roundtrip(self):
        text = to_csv(build_characteristics(make_pmi()))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[1]["specification"] == "4X ⌀6 ±0.15"
        assert rows[1]["face_ids"] == "101 102"

    def test_csv_german_excel(self):
        text = to_csv(build_characteristics(make_pmi()), delimiter=";", decimal_comma=True)
        assert "5,85" in text and text.splitlines()[0].startswith("number;")

    def test_csv_file_has_bom(self, tmp_path):
        out = tmp_path / "chars.csv"
        to_csv(build_characteristics(make_pmi()), str(out))
        assert out.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_json_includes_warnings(self):
        pmi = make_pmi()
        pmi.warnings = ["something odd"]
        data = json.loads(to_json(build_characteristics(pmi), pmi, source="x.stp"))
        assert data["count"] == 6 and data["warnings"] == ["something odd"]

    def test_table(self):
        assert "Specification" in to_table(build_characteristics(make_pmi()))


@nist_mark
def test_nist_ftc10_characteristics():
    pmi = cadtokens.extract_pmi(cadtokens.load(str(NIST_DIR / "nist_ftc_10_asme1_ap242-e2.stp")))
    specs = [c.specification for c in build_characteristics(pmi)]
    assert len(specs) == 66
    for expected in ("8X ⌖ | ⌀0.75 | A | B | C", "2X ⌖ | ⌀0.8 Ⓟ3.2 | D | E | B",
                     "3X ⌖ | ⌀0 Ⓛ | H | K | L", "⏤ | 0.2/15",
                     "⌀3.50 G6 (+0.012/+0.004)"):
        assert expected in specs
