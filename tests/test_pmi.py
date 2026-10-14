"""Tests for PMI (GD&T) extraction."""

import cadtokens
from cadtokens.step_engine.pmi import (
    PMIExtractor, PMIWriter, PMIData,
    ToleranceInfo, DimensionInfo, DatumInfo, DatumSystemInfo,
    extract_pmi, tolerance_category,
)


class TestPMIDataModel:
    def test_pmi_data_creation(self):
        pmi = PMIData(
            tolerances=[],
            dimensions=[],
            datums=[],
            datum_systems=[],
        )
        assert len(pmi.tolerances) == 0
        assert len(pmi.dimensions) == 0

    def test_tolerance_info(self):
        t = ToleranceInfo(
            entity_id=100,
            name="test_tol",
            tolerance_type="POSITION",
            value=0.02,
            face_ids=[14, 22],
            datum_labels=["A", "B"],
        )
        assert t.tolerance_type == "POSITION"
        assert t.value == 0.02
        assert len(t.face_ids) == 2
        assert len(t.datum_labels) == 2

    def test_dimension_info(self):
        d = DimensionInfo(
            entity_id=200,
            name="test_dim",
            dimension_type="DIMENSIONAL_SIZE",
            nominal_value=12.0,
            face_ids=[14],
        )
        assert d.dimension_type == "DIMENSIONAL_SIZE"
        assert d.nominal_value == 12.0
        assert len(d.face_ids) == 1

    def test_datum_info(self):
        d = DatumInfo(entity_id=300, label="A", face_ids=[7])
        assert d.label == "A"
        assert d.face_ids == [7]


class TestPMIExtraction:
    def test_extract_from_sample(self, step_file):
        """Sample file may not have PMI, but extraction should not crash."""
        pmi = extract_pmi(step_file)
        assert isinstance(pmi, PMIData)
        assert isinstance(pmi.tolerances, list)
        assert isinstance(pmi.dimensions, list)
        assert isinstance(pmi.datums, list)

    def test_extractor_class(self, step_file):
        import cadtokens
        g = cadtokens.graph(step_file)
        ext = PMIExtractor(g)
        pmi = ext.extract()
        assert isinstance(pmi, PMIData)

    def test_tolerance_category_function(self):
        cat = tolerance_category("POSITION")
        assert isinstance(cat, str)
        assert cat == "location"

        cat2 = tolerance_category("FLATNESS")
        assert cat2 == "form"
