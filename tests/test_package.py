"""Tests for the top-level cadtokens package API."""

import cadtokens


class TestPackageAPI:
    def test_version(self):
        assert cadtokens.__version__ == "0.2.0"

    def test_core_exports(self):
        """All core symbols should be importable."""
        assert callable(cadtokens.load)
        assert callable(cadtokens.save)
        assert callable(cadtokens.edit)
        assert callable(cadtokens.graph)
        assert callable(cadtokens.parse)
        assert callable(cadtokens.parse_file)
        assert callable(cadtokens.serialize)
        assert callable(cadtokens.extract_pmi)

    def test_classes_importable(self):
        assert cadtokens.StepFile is not None
        assert cadtokens.Entity is not None
        assert cadtokens.Ref is not None
        assert cadtokens.ReferenceGraph is not None
        assert cadtokens.BRepExtractor is not None
        assert cadtokens.StepEditor is not None

    def test_pmi_classes_importable(self):
        assert cadtokens.PMIExtractor is not None
        assert cadtokens.PMIWriter is not None
        assert cadtokens.PMIData is not None
        assert cadtokens.ToleranceInfo is not None
        assert cadtokens.DimensionInfo is not None
        assert cadtokens.DatumInfo is not None
