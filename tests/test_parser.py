"""Tests for the STEP parser — the core, zero-dependency feature."""

import cadtokens
from cadtokens.step_engine.model import StepFile, Entity, Ref, EnumValue, TypedValue
from cadtokens.step_engine.tokenizer import tokenize
from cadtokens.step_engine.parser import parse, parse_file, ParseError


# ── Tokenizer ───────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_tokenize_entity(self):
        tokens = tokenize("#1=POINT('origin',1.0);")
        names = [t.type for t in tokens]
        assert "ENTITY_REF" in names
        assert "KEYWORD" in names
        assert "STRING" in names

    def test_tokenize_empty_data(self):
        tokens = tokenize("DATA;\nENDSEC;")
        assert len(tokens) >= 2

    def test_entity_ref_value(self):
        tokens = tokenize("#42=FOO();")
        refs = [t for t in tokens if t.type == "ENTITY_REF"]
        assert len(refs) == 1
        assert refs[0].value == "#42"


# ── Parser ──────────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_minimal_step(self):
        """Parse a minimal valid STEP string."""
        text = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('test'),'2;1');\n"
            "FILE_NAME('test','2026-01-01',(''),(''),'','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
            "ENDSEC;\n"
            "DATA;\n"
            "#1=CARTESIAN_POINT('origin',(0.0,0.0,0.0));\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n"
        )
        sf = parse(text)
        assert isinstance(sf, StepFile)
        assert len(sf) == 1
        assert 1 in sf

    def test_parse_entity_params(self):
        text = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('test'),'2;1');\n"
            "FILE_NAME('test','2026-01-01',(''),(''),'','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
            "ENDSEC;\n"
            "DATA;\n"
            "#1=CARTESIAN_POINT('origin',(1.5,2.5,3.5));\n"
            "#2=DIRECTION('',(0.0,0.0,1.0));\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n"
        )
        sf = parse(text)
        assert len(sf) == 2
        pt = sf[1]
        assert pt.type_name == "CARTESIAN_POINT"
        assert pt.params[0] == "origin"

    def test_parse_refs(self):
        text = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('test'),'2;1');\n"
            "FILE_NAME('test','2026-01-01',(''),(''),'','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
            "ENDSEC;\n"
            "DATA;\n"
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));\n"
            "#2=VERTEX_POINT('',#1);\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n"
        )
        sf = parse(text)
        vp = sf[2]
        # params[1] should be a Ref to #1
        assert isinstance(vp.params[1], Ref)
        assert vp.params[1].entity_id == 1


# ── Load + Parse file ──────────────────────────────────────────────────────

class TestLoadFile:
    def test_load_sample(self, sample_step_path):
        sf = cadtokens.load(sample_step_path)
        assert isinstance(sf, StepFile)
        assert len(sf) == 628

    def test_parse_file(self, sample_step_path):
        sf = parse_file(sample_step_path)
        assert len(sf) == 628

    def test_header_exists(self, step_file):
        assert step_file.header is not None
        assert step_file.header.file_schema is not None


# ── Reference Graph ─────────────────────────────────────────────────────────

class TestReferenceGraph:
    def test_entity_count_by_type(self, ref_graph):
        counts = ref_graph.entity_count_by_type()
        assert isinstance(counts, dict)
        assert "CARTESIAN_POINT" in counts
        assert counts["CARTESIAN_POINT"] > 0

    def test_find_by_type(self, ref_graph):
        faces = ref_graph.find_by_type("ADVANCED_FACE")
        assert len(faces) == 11

    def test_references(self, ref_graph):
        # First entity should have refs (or at least not crash)
        face_ids = ref_graph.find_by_type("ADVANCED_FACE")
        refs = ref_graph.references(face_ids[0])
        assert isinstance(refs, set)

    def test_roots_and_leaves(self, ref_graph):
        roots = ref_graph.roots()
        leaves = ref_graph.leaves()
        assert len(roots) > 0
        assert len(leaves) > 0

    def test_summary(self, ref_graph):
        s = ref_graph.summary()
        assert "entities" in s
        assert s["entities"] == 628


# ── BRepExtractor ───────────────────────────────────────────────────────────

class TestBRepExtractor:
    def test_faces(self, brep):
        faces = brep.get_faces()
        assert len(faces) == 11
        assert all(isinstance(f, int) for f in faces)

    def test_edge_curves(self, brep):
        edges = brep.get_edge_curves()
        assert len(edges) == 21

    def test_vertices(self, brep):
        verts = brep.get_vertices()
        assert len(verts) == 12

    def test_points(self, brep):
        pts = brep.get_points()
        assert len(pts) == 89
        # Each point is (x, y, z)
        for eid, coords in list(pts.items())[:3]:
            assert len(coords) == 3
            assert all(isinstance(c, float) for c in coords)

    def test_face_surface(self, brep):
        faces = brep.get_faces()
        surf = brep.get_face_surface(faces[0])
        assert surf is not None
        assert surf.type_name == "PLANE"


# ── Serializer ──────────────────────────────────────────────────────────────

class TestSerializer:
    def test_serialize_roundtrip(self, step_file):
        text = cadtokens.serialize(step_file)
        assert isinstance(text, str)
        assert len(text) > 1000
        assert "ISO-10303-21" in text

        # Re-parse the serialized output
        sf2 = cadtokens.parse(text)
        assert len(sf2) == len(step_file)

    def test_save_and_reload(self, step_file, tmp_path):
        out_path = str(tmp_path / "roundtrip.step")
        cadtokens.save(step_file, out_path)

        sf2 = cadtokens.load(out_path)
        assert len(sf2) == len(step_file)


# ── Editor ──────────────────────────────────────────────────────────────────

class TestEditor:
    def test_translate(self, sample_step_path):
        sf = cadtokens.load(sample_step_path)
        g = cadtokens.graph(sf)
        brep = cadtokens.BRepExtractor(g)
        pts_before = brep.get_points()
        first_id = sorted(pts_before.keys())[0]
        x0, y0, z0 = pts_before[first_id]

        editor = cadtokens.edit(sf)
        editor.translate(100.0, 200.0, 300.0)

        g2 = cadtokens.graph(sf)
        brep2 = cadtokens.BRepExtractor(g2)
        pts_after = brep2.get_points()
        x1, y1, z1 = pts_after[first_id]

        assert abs(x1 - (x0 + 100.0)) < 1e-6
        assert abs(y1 - (y0 + 200.0)) < 1e-6
        assert abs(z1 - (z0 + 300.0)) < 1e-6

    def test_renumber(self, sample_step_path):
        sf = cadtokens.load(sample_step_path)
        editor = cadtokens.edit(sf)
        editor.renumber()
        # After renumber, entity IDs should be contiguous 1..N
        ids = sorted(sf.entities.keys())
        assert ids[0] == 1
        assert ids[-1] == len(sf)
