"""Tests for the MCP server tools (tool functions need no MCP SDK)."""

import asyncio
import json

import pytest

from cadtokens import mcp_server as srv
from tests.test_nist import NIST_DIR, pytestmark as nist_mark

FTC10 = str(NIST_DIR / "nist_ftc_10_asme1_ap242-e2.stp")


def test_rejects_non_step_paths(tmp_path):
    bad = tmp_path / "secrets.txt"
    bad.write_text("x")
    with pytest.raises(ValueError):
        srv.pmi_summary(str(bad))
    with pytest.raises(FileNotFoundError):
        srv.pmi_summary(str(tmp_path / "missing.stp"))


def test_sample_without_pmi(sample_step_path):
    summary = srv.pmi_summary(sample_step_path)
    assert summary["has_semantic_pmi"] is False
    assert srv.list_characteristics(sample_step_path)["count"] == 0


@nist_mark
class TestOnNIST:
    def test_summary(self):
        s = srv.pmi_summary(FTC10)
        assert s["length_unit"] == "mm" and s["geometric_tolerances"] == 40
        assert "D | E-F-G" in s["datum_reference_frames"]

    def test_filters(self):
        tight = srv.list_characteristics(FTC10, kind="geometric_tolerance", max_tolerance=0.15)
        assert tight["count"] == 10
        assert all(c["max"] <= 0.15 for c in tight["characteristics"])

    def test_get_characteristic_and_face_lookup(self):
        first_face = srv.list_characteristics(FTC10)["characteristics"][2]["faces"][0]
        lookup = srv.face_pmi(FTC10, first_face)
        assert any(c["specification"].startswith("⌀6") for c in lookup["characteristics"])
        assert srv.get_characteristic(FTC10, 1)["number"] == 1
        with pytest.raises(ValueError):
            srv.get_characteristic(FTC10, 999)

    def test_datum_frames(self):
        frames = srv.datum_reference_frames(FTC10)
        assert {d["label"] for d in frames["datums"]} >= {"A", "B", "C"}
        assert any(f["frame"].startswith("D | E-F-G") for f in frames["frames"])

    def test_export_refuses_overwrite(self, tmp_path):
        out = tmp_path / "chars.csv"
        assert srv.export_characteristics(FTC10, str(out))["count"] == 66
        with pytest.raises(FileExistsError):
            srv.export_characteristics(FTC10, str(out))
        srv.export_characteristics(FTC10, str(out), format="json", overwrite=True)
        assert json.loads(out.read_text())["count"] == 66


def test_server_registers_tools():
    pytest.importorskip("mcp")
    server = srv.build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"pmi_summary", "list_characteristics", "get_characteristic",
                     "face_pmi", "datum_reference_frames", "export_characteristics"}
