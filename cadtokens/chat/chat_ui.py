#!/usr/bin/env python3
"""
CAD-Brain Chat v2 — GNN-Enhanced CAD Analysis

Combines:
1. OCC Tools — exact measurements (volume, area, bounding box, radii)
2. BRepGNN — deep semantic analysis (face classification, feature detection, similarity)
3. Ollama LLM — natural language conversation about CAD parts
4. 3D Viewer — interactive visualization with face highlighting

Usage:
    python scripts/cad_chat_v2.py [--port 7860] [--model ministral-3:14b]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import json
import time
import re
import tempfile
import base64
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


import gradio as gr
import ollama

from cadtokens.tools.geometry_tools import GeometryTools, TOOL_DEFINITIONS
from cadtokens.tools.cad_renderer import CADRenderer
from cadtokens.gnn.gnn_tools import GNNTools, GNN_TOOL_DEFINITIONS


# =============================================================================
#  Globals
# =============================================================================

occ_tools = GeometryTools()
renderer = CADRenderer()
gnn_tools = GNNTools()

current_step_path: Optional[str] = None
current_model_path: Optional[str] = None
current_shape = None  # OCC TopoDS_Shape (for GNN)

CURRENT_MODEL = "ministral-3:14b"


# =============================================================================
#  System Prompt
# =============================================================================

SYSTEM_PROMPT = """Du bist ein fortschrittlicher CAD-Analyse-Experte mit Zugriff auf zwei Arten von Analyse-Tools:

## 1. OCC Tools (exakte Geometrie-Messung)
- get_bounding_box: Maße des Bauteils (Breite, Höhe, Tiefe in mm)
- get_volume_and_area: Volumen und Oberfläche
- count_features: Zählt Flächentypen (Plane, Cylinder, Cone, Sphere, Torus, BSpline)
- list_faces: Alle Faces mit ID, Typ, Fläche
- get_face_info: Details zu einer Face (Typ, Radius, Normal, Maße)
- get_hole_diameters: Findet Bohrungen mit Durchmesser und passender Schraubengröße
- get_summary: Komplette geometrische Übersicht

## 2. GNN Tools (KI-basierte Feature-Erkennung)
Diese Tools nutzen ein trainiertes Graph Neural Network (BRepGNN) mit 12-Klassen semantischer Segmentierung:
- Erkennt: stock_wall, stock_cap (Grundkörper), hole_wall, hole_floor (Bohrungen),
  pocket_wall, pocket_floor (Taschen), boss_wall, boss_cap (Erhebungen),
  fillet (Verrundungen), chamfer (Fasen), revolve_body (Drehteile)
- gnn_smart_summary: Intelligente Zusammenfassung mit Feature-Erkennung
- gnn_detect_features: Erkennt Manufacturing Features und gruppiert Faces nach Rolle
- gnn_classify_faces: Klassifiziert alle Faces mit KI-Konfidenz
- gnn_find_similar_faces: Findet ähnliche Faces über Embedding-Distanz
- gnn_face_info: Detaillierte KI-Analyse einer einzelnen Face
- gnn_shape_fingerprint: Globaler Shape-Fingerabdruck

## 3. Vision
- look_at_model: Rendert Bilder des Bauteils aus verschiedenen Blickwinkeln

WICHTIGE REGELN:
1. Nutze OCC-Tools für exakte Maße (mm) und GNN-Tools für Feature-Erkennung
2. Für eine Übersicht des Bauteils: kombiniere get_summary UND gnn_smart_summary
3. Wenn nach Features gefragt wird (Bohrungen, Taschen, etc.): nutze gnn_detect_features
4. Antworte auf Deutsch, präzise und technisch
5. Erkläre was die GNN-Analyse bedeutet (z.B. "Das GNN hat 3 Durchgangsbohrungen erkannt")
6. Bei niedrigem GNN-Konfidenz (<60%): weise darauf hin
7. Nutze look_at_model um das Bauteil visuell zu inspizieren wenn nötig"""


# =============================================================================
#  Tool Registration
# =============================================================================

def get_all_tool_defs() -> List[dict]:
    """Build Ollama-compatible tool definitions."""
    tools_list = []

    # OCC tools
    for tool_def in TOOL_DEFINITIONS:
        params = tool_def["parameters"].copy()
        props = {k: v for k, v in params["properties"].items() if k != "step_path"}
        required = [r for r in params.get("required", []) if r != "step_path"]
        tools_list.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })

    # GNN tools
    for tool_def in GNN_TOOL_DEFINITIONS:
        params = tool_def["parameters"].copy()
        props = {k: v for k, v in params["properties"].items() if k != "step_path"}
        required = [r for r in params.get("required", []) if r != "step_path"]
        tools_list.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })

    # Vision tool
    tools_list.append({
        "type": "function",
        "function": {
            "name": "look_at_model",
            "description": "Rendert Bilder des CAD-Bauteils aus verschiedenen Blickwinkeln.",
            "parameters": {
                "type": "object",
                "properties": {
                    "views": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["front", "back", "top", "bottom", "left", "right", "iso"]},
                        "description": "Liste von 1-3 Blickwinkeln",
                    }
                },
                "required": [],
            },
        },
    })

    return tools_list


def execute_tool(tool_name: str, arguments: dict) -> Any:
    """Route tool call to OCC or GNN backend."""
    global current_shape

    # OCC tools
    if hasattr(occ_tools, tool_name):
        arguments["step_path"] = current_step_path
        try:
            return getattr(occ_tools, tool_name)(**arguments)
        except Exception as e:
            return {"error": str(e)}

    # GNN tools
    gnn_method_map = {
        "gnn_classify_faces": "classify_faces",
        "gnn_detect_features": "detect_features",
        "gnn_find_similar_faces": "find_similar_faces",
        "gnn_face_info": "get_face_embedding_info",
        "gnn_smart_summary": "get_smart_summary",
        "gnn_shape_fingerprint": "get_shape_fingerprint",
    }

    if tool_name in gnn_method_map:
        if current_shape is None:
            return {"error": "Kein Bauteil geladen"}
        method_name = gnn_method_map[tool_name]
        method = getattr(gnn_tools, method_name)
        # Pass step_path (not shape) for training-consistent feature extraction
        arguments.pop("step_path", None)
        try:
            return method(current_step_path, **arguments)
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unbekanntes Tool: {tool_name}"}


def execute_look_at_model(views: List[str] = None) -> List[str]:
    """Render part images for vision."""
    if not current_step_path:
        return []
    if not views:
        views = ["iso", "front", "top"]
    views = views[:3]
    try:
        image_paths = renderer.render_to_image(
            current_step_path, size=(512, 512), views=views,
        )
        return [
            base64.b64encode(open(p, "rb").read()).decode("utf-8")
            for p in image_paths
        ]
    except Exception as e:
        print(f"  Vision error: {e}", flush=True)
        return []


# =============================================================================
#  Chat
# =============================================================================

def chat_with_cad(message: str, history: list) -> Tuple[str, Optional[List[int]]]:
    """Chat with tool calling. Returns (response, highlight_face_ids)."""
    global current_step_path

    if not current_step_path:
        return "❌ Bitte lade zuerst eine STEP-Datei.", None

    t0 = time.time()

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history or []:
        if isinstance(h, dict):
            role = h.get("role", "user")
            content = h.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    tool_defs = get_all_tool_defs()
    tool_log = []
    used_faces = set()

    for iteration in range(5):  # Max 5 tool rounds
        response = ollama.chat(
            model=CURRENT_MODEL,
            messages=messages,
            tools=tool_defs,
            options={"temperature": 0.1},
        )

        assistant_msg = response["message"]
        messages.append(assistant_msg)

        if not assistant_msg.get("tool_calls"):
            break

        for tool_call in assistant_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            # Collect face IDs
            for key in ("face_id", "face1_id", "face2_id"):
                if key in args:
                    used_faces.add(args[key])

            t1 = time.time()

            if tool_name == "look_at_model":
                views = args.get("views", ["iso", "front", "top"])
                b64_images = execute_look_at_model(views)
                tool_log.append(f"👁️ look_at_model ({', '.join(views[:len(b64_images)])})")
                messages.append({
                    "role": "tool",
                    "content": f"Hier sind {len(b64_images)} Ansichten: {', '.join(views[:len(b64_images)])}. Beschreibe was du siehst.",
                    "images": b64_images,
                })
                print(f"  👁️ look_at_model: {len(b64_images)} images ({time.time()-t1:.1f}s)", flush=True)
                continue

            result = execute_tool(tool_name, args)
            dt = time.time() - t1

            # Collect face IDs from results
            if isinstance(result, dict):
                for f in result.get("faces", [])[:10]:
                    if isinstance(f, dict) and "face_id" in f:
                        used_faces.add(f["face_id"])
                # From GNN similar faces
                for f in result.get("similar_faces", []):
                    if isinstance(f, dict) and "face_id" in f:
                        used_faces.add(f["face_id"])
                # From GNN features
                for feat in result.get("features", []):
                    if isinstance(feat, dict):
                        for fid in feat.get("face_ids", [])[:5]:
                            used_faces.add(fid)

            icon = "🧠" if tool_name.startswith("gnn_") else "🔧"
            tool_log.append(f"{icon} {tool_name} ({dt:.1f}s)")
            print(f"  {icon} {tool_name}: {dt:.2f}s", flush=True)

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # Extract response
    final = assistant_msg.get("content", "")

    # Extract face IDs mentioned in response
    for fid in re.findall(r"[Ff]ace\s*#?\s*(\d+)", final):
        used_faces.add(int(fid))

    if tool_log:
        final = " | ".join(tool_log) + "\n\n" + final

    print(f"  Total: {time.time()-t0:.1f}s, faces: {sorted(used_faces)}", flush=True)
    return final, list(used_faces) if used_faces else None


# =============================================================================
#  File Loading
# =============================================================================

def load_step_file(file) -> Tuple[Optional[str], str, str]:
    """Load STEP → 3D preview + GNN analysis + status. Returns (model_path, analysis_md, status)."""
    global current_step_path, current_model_path, current_shape

    if file is None:
        return None, "", "❌ Keine Datei"

    try:
        step_path = str(file)
        current_step_path = step_path

        # Load shape for GNN
        from OCC.Core.STEPControl import STEPControl_Reader
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != 1:
            return None, "", f"❌ Konnte STEP nicht laden"
        reader.TransferRoots()
        current_shape = reader.OneShape()
        gnn_tools.clear_cache()

        # 3D conversion
        try:
            model_path = renderer.step_to_glb(step_path)
        except Exception:
            model_path = renderer.step_to_stl(step_path)
        current_model_path = model_path

        # Quick OCC summary
        occ_summary = occ_tools.get_summary(step_path)
        nf = occ_summary["topology"]["faces"]
        ne = occ_summary["topology"]["edges"]
        vol = occ_summary["properties"]["volume"]
        area = occ_summary["properties"]["surface_area"]
        bb = occ_summary.get("dimensions", occ_summary.get("properties", {}).get("bounding_box", {}))

        # Quick GNN summary (pass step_path for training-consistent features)
        gnn_summary = gnn_tools.get_smart_summary(step_path)
        ft = gnn_summary.get("face_types", {})
        seg = gnn_summary.get("feature_segments", {})
        conf = gnn_summary.get("confidence", {})

        # Build analysis markdown
        analysis = f"""### Geometrie (OCC)
| | |
|---|---|
| **Faces / Edges** | {nf} / {ne} |
| **Volumen** | {vol:.1f} mm³ |
| **Oberfläche** | {area:.1f} mm² |
| **Bounding Box** | {bb['width']:.1f} × {bb['height']:.1f} × {bb['depth']:.1f} mm |

### KI-Analyse (BRepGNN)
| | |
|---|---|
| **Flächentypen** | {', '.join(f'{v}× {k}' for k,v in ft.items())} |
| **Feature-Segmente** | {', '.join(f'{v}× {k}' for k,v in seg.items())} |
| **GNN Konfidenz** | Typ: {conf.get('face_type_avg', 0):.0%} · Segment: {conf.get('segment_avg', 0):.0%} |
"""
        # Detected features (semantic 12-class labels)
        holes = gnn_summary.get("holes", [])
        pockets = gnn_summary.get("pockets", [])
        bosses = gnn_summary.get("bosses", [])
        fillets = gnn_summary.get("detected_fillets", [])
        chamfers = gnn_summary.get("detected_chamfers", [])
        revolves = gnn_summary.get("revolve_features", [])

        detected = []
        if holes:
            detected.append(f"🔘 {len(holes)} Bohrungsfläche(n)")
        if pockets:
            detected.append(f"▫ {len(pockets)} Taschenfläche(n)")
        if bosses:
            detected.append(f"🔲 {len(bosses)} Boss-Fläche(n)")
        if fillets:
            detected.append(f"↪ {len(fillets)} Verrundung(en)")
        if chamfers:
            detected.append(f"◇ {len(chamfers)} Fase(n)")
        if revolves:
            detected.append(f"🔄 {len(revolves)} Rotationsfläche(n)")
        if detected:
            analysis += f"\n**Erkannte Features:** {' · '.join(detected)}"

        status_text = f"✅ {Path(step_path).name} — {nf} Faces, GNN Typ-Konf. {conf.get('face_type_avg', 0):.0%}"
        return model_path, analysis, status_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, "", f"❌ Fehler: {e}"


# =============================================================================
#  Gradio UI
# =============================================================================

def build_ui():
    with gr.Blocks(title="CAD-Brain Chat v2") as demo:
        gr.Markdown(
            """
            # 🧠 CAD-Brain Chat v2
            **OCC Geometrie-Tools + BRepGNN Feature-Erkennung + LLM Chat**

            Upload STEP → automatische Analyse (Geometrie + KI) → Chat mit dem CAD-Experten
            """
        )

        with gr.Row():
            status = gr.Markdown("⏳ Lade GNN …")

        with gr.Row():
            # ─── Left: 3D + Upload ───────────────────────────────
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="STEP-Datei hochladen",
                    file_types=[".step", ".stp", ".STEP", ".STP"],
                    type="filepath",
                )

                model_viewer = gr.Model3D(label="3D Ansicht", height=380)

                analysis_panel = gr.Markdown("", label="Analyse")

                # Example buttons
                gr.Markdown("**Beispiele:**")
                with gr.Row():
                    example_paths = _find_examples()
                    example_btns = []
                    for name, path in example_paths[:4]:
                        btn = gr.Button(name, size="sm", variant="secondary")
                        example_btns.append((btn, path))

            # ─── Right: Chat ─────────────────────────────────────
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chat", height=400)

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Frage",
                        placeholder="z.B. Welche Features hat das Bauteil?",
                        scale=4,
                    )
                    send_btn = gr.Button("Senden", variant="primary", scale=1)

                gr.Markdown("**Schnellfragen:**")
                with gr.Row():
                    q_btns = [
                        ("📊 Zusammenfassung", "Gib mir eine vollständige Zusammenfassung. Nutze sowohl OCC als auch GNN Analyse."),
                        ("🔍 Features?", "Welche Manufacturing Features hat das Bauteil? Nutze gnn_detect_features."),
                        ("📏 Maße?", "Wie groß ist das Bauteil? Gib mir die genauen Maße."),
                        ("🔘 Bohrungen?", "Gibt es Bohrungen? Nutze get_hole_diameters UND gnn_detect_features."),
                    ]
                    quick_btns = []
                    for label, _ in q_btns:
                        quick_btns.append(gr.Button(label, size="sm"))

                with gr.Row():
                    q2_btns = [
                        ("🧠 GNN Details", "Zeige mir die GNN Smart Summary mit allen erkannten Features."),
                        ("🔗 Ähnliche Faces", "Welche Faces sind Face 0 am ähnlichsten? Nutze gnn_find_similar_faces."),
                        ("🎯 Face 0 Info", "Analysiere Face 0 detailliert mit gnn_face_info."),
                        ("🗑 Chat löschen", None),
                    ]
                    quick_btns2 = []
                    for label, _ in q2_btns:
                        quick_btns2.append(gr.Button(label, size="sm"))

        # ─── Event handlers ──────────────────────────────────────

        def on_file_upload(file):
            model_path, analysis, status_text = load_step_file(file)
            return model_path, analysis, status_text

        def on_send(message, history):
            if not message.strip():
                return "", history, gr.update()

            response, highlight_faces = chat_with_cad(message, history or [])

            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

            new_model = gr.update()
            if highlight_faces and current_step_path:
                try:
                    new_model = renderer.step_to_glb(
                        current_step_path,
                        highlight_faces=highlight_faces,
                        highlight_color=(255, 80, 80, 255),
                        base_color=(180, 180, 190, 255),
                    )
                except:
                    pass

            return "", history, new_model

        def reset_model():
            if current_step_path:
                try:
                    return renderer.step_to_glb(current_step_path)
                except:
                    pass
            return gr.update()

        # File upload
        file_input.change(
            on_file_upload,
            inputs=[file_input],
            outputs=[model_viewer, analysis_panel, status],
        )

        # Example buttons
        for btn, path in example_btns:
            def make_cb(p):
                def cb():
                    return p
                return cb
            btn.click(fn=make_cb(path), outputs=[file_input])

        # Chat
        send_btn.click(on_send, [msg_input, chatbot], [msg_input, chatbot, model_viewer])
        msg_input.submit(on_send, [msg_input, chatbot], [msg_input, chatbot, model_viewer])

        # Quick question buttons
        for btn, (label, question) in zip(quick_btns, q_btns):
            if question:
                btn.click(
                    lambda h, q=question: on_send(q, h),
                    inputs=[chatbot],
                    outputs=[msg_input, chatbot, model_viewer],
                )

        for btn, (label, question) in zip(quick_btns2, q2_btns):
            if question:
                btn.click(
                    lambda h, q=question: on_send(q, h),
                    inputs=[chatbot],
                    outputs=[msg_input, chatbot, model_viewer],
                )
            else:
                # Clear button
                btn.click(lambda: ([], reset_model()), outputs=[chatbot, model_viewer])

        # Startup
        def on_load():
            return "✅ GNN geladen — STEP-Datei hochladen zum Starten"
        demo.load(fn=on_load, outputs=[status])

    return demo


def _find_examples() -> List[Tuple[str, str]]:
    """Find example STEP files for the UI."""
    examples = []

    # s2.0.1 dataset
    d = Path("data/s2.0.1/breps/step")
    if d.exists():
        for f in sorted(d.glob("*.stp"))[:2]:
            examples.append((f.stem[:20], str(f)))

    # Extrude tools
    d = Path("data/r1.0.1_extrude_tools/extrude_tools")
    if d.exists():
        step_files = sorted(d.glob("*.step"))
        if step_files:
            examples.append(("ExtrTool1", str(step_files[0])))
            if len(step_files) > 10:
                examples.append(("ExtrTool2", str(step_files[10])))

    # NIST
    import glob
    nist = sorted(glob.glob("data/nist/**/*.stp", recursive=True))
    if nist:
        examples.append(("NIST", nist[0]))

    # Test files
    for f in Path(".").glob("*.step"):
        examples.append((f.stem[:20], str(f)))

    return examples[:6]


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAD-Brain Chat v2")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", type=str, default="ministral-3:14b")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    CURRENT_MODEL = args.model

    # Load GNN at startup
    print("=" * 60)
    print("  CAD-Brain Chat v2 — Loading pipeline")
    print("=" * 60)
    t0 = time.time()
    ok = gnn_tools.load()
    print(f"  GNN loaded: {ok} ({time.time()-t0:.1f}s)")
    print(f"  LLM: {CURRENT_MODEL} (Ollama)")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
