#!/usr/bin/env python3
"""
CAD Chat Agent - Phase 1

ReAct Agent der mit CAD-Dateien "reden" kann.
Nutzt Ollama (qwen3) mit Function Calling.

Usage:
    python src/agent/cad_chat.py data/s2.0.1/breps/step/100027_258e3965_0.stp
    
    # Oder interaktiv:
    python src/agent/cad_chat.py --interactive
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ollama
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("⚠️  ollama nicht installiert. Installiere mit: pip install ollama")

# Eigene Tools
from cadtokens.tools.geometry_tools import GeometryTools, TOOL_DEFINITIONS


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """Du bist ein CAD-Experte und Ingenieur. Du analysierst STEP-Dateien präzise und hilfst bei technischen Fragen.

WICHTIGE REGELN:
1. RATE NICHT. Nutze IMMER die verfügbaren Tools um Geometrie zu inspizieren.
2. Antworte auf Deutsch, präzise und technisch korrekt.
3. Gib Maße mit Einheiten an (die Werte sind in mm).
4. Wenn du mehrere Informationen brauchst, rufe mehrere Tools auf.

Die aktuelle STEP-Datei ist: {step_path}

Du hast Zugriff auf folgende Tools zur Analyse:
- get_bounding_box: Maße des Bauteils (Breite, Höhe, Tiefe)
- get_volume_and_area: Volumen und Oberfläche
- count_features: Zählt geometrische Features (Plane, Cylinder, etc.)
- list_faces: Listet alle Faces mit ID, Typ, Fläche
- get_face_info: Details zu einer bestimmten Face
- measure_distance: Abstand zwischen zwei Faces
- check_parallel: Prüft Parallelität zweier Faces
- get_summary: Komplette Übersicht des Bauteils

Beantworte die Fragen des Users basierend auf den Tool-Ergebnissen."""


# =============================================================================
# Tool Definitions für Ollama
# =============================================================================

def get_ollama_tools(step_path: str) -> List[Dict]:
    """Konvertiert Tool-Definitionen für Ollama Format"""
    tools = []
    
    for tool_def in TOOL_DEFINITIONS:
        # Entferne step_path aus required - wir setzen es automatisch
        params = tool_def["parameters"].copy()
        props = params["properties"].copy()
        
        # step_path entfernen (wird automatisch gesetzt)
        if "step_path" in props:
            del props["step_path"]
        
        required = [r for r in params.get("required", []) if r != "step_path"]
        
        tools.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        })
    
    return tools


# =============================================================================
# CAD Chat Agent
# =============================================================================

# Runs fully local. Ollama models with a "-cloud" suffix are executed on
# Ollama's servers — the STEP data would leave the machine.
DEFAULT_MODEL = "qwen3:14b"


def warn_if_cloud_model(model: str):
    if model.endswith("-cloud") or ":cloud" in model:
        print(f"⚠️  '{model}' is an Ollama cloud model: tool results (your CAD data) "
              f"are sent to Ollama's servers. Use a local model to keep data on-prem.",
              file=sys.stderr)

class CADChatAgent:
    """
    ReAct Agent für CAD-Analyse mit Ollama.
    """
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL,
                 step_path: Optional[str] = None):
        """
        Args:
            model: Ollama Modell (z.B. qwen3:32b, qwen3:14b)
            step_path: Pfad zur STEP-Datei
        """
        self.model = model
        self.step_path = step_path
        warn_if_cloud_model(model)
        self.tools = GeometryTools()
        self.conversation: List[Dict] = []
        
        if not HAS_OLLAMA:
            raise RuntimeError("Ollama nicht installiert!")
    
    def set_step_file(self, step_path: str):
        """Setzt die aktuelle STEP-Datei"""
        if not Path(step_path).exists():
            raise FileNotFoundError(f"STEP-Datei nicht gefunden: {step_path}")
        self.step_path = step_path
        self.tools._cache.clear()  # Cache leeren
        self.tools._face_cache.clear()
        print(f"📁 Arbeite mit: {Path(step_path).name}")
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Führt ein Tool aus"""
        # step_path automatisch hinzufügen
        arguments["step_path"] = self.step_path
        
        # Tool aufrufen
        tool_method = getattr(self.tools, tool_name, None)
        if tool_method is None:
            return {"error": f"Tool '{tool_name}' nicht gefunden"}
        
        try:
            result = tool_method(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Verarbeitet eine User-Nachricht mit ReAct Loop.
        
        Args:
            user_message: Frage des Users
            max_iterations: Max Anzahl Tool-Aufrufe
            
        Returns:
            Antwort des Agents
        """
        if not self.step_path:
            return "❌ Keine STEP-Datei geladen. Nutze set_step_file() zuerst."
        
        # System Prompt mit aktuellem Pfad
        system_msg = SYSTEM_PROMPT.format(step_path=Path(self.step_path).name)
        
        # Conversation aufbauen
        messages = [
            {"role": "system", "content": system_msg}
        ]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_message})
        
        # Tools für Ollama
        ollama_tools = get_ollama_tools(self.step_path)
        
        # ReAct Loop
        for iteration in range(max_iterations):
            # LLM aufrufen
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=ollama_tools,
                options={"temperature": 0.1}  # Deterministischer für Tools
            )
            
            assistant_message = response["message"]
            messages.append(assistant_message)
            
            # Keine Tool Calls? Dann fertig
            if not assistant_message.get("tool_calls"):
                # Conversation speichern
                self.conversation.append({"role": "user", "content": user_message})
                self.conversation.append(assistant_message)
                return assistant_message.get("content", "")
            
            # Tools ausführen
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                
                # Parse arguments (kann String oder Dict sein)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except:
                        arguments = {}
                
                print(f"   🔧 {tool_name}({arguments})")
                
                # Tool ausführen
                result = self._execute_tool(tool_name, arguments)
                
                # Result als Tool-Antwort hinzufügen
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        return "⚠️ Maximale Iterationen erreicht."
    
    def reset_conversation(self):
        """Setzt Conversation zurück"""
        self.conversation = []


# =============================================================================
# Interactive CLI
# =============================================================================

def interactive_mode(agent: CADChatAgent):
    """Interaktiver Chat-Modus"""
    print("\n" + "="*60)
    print("🔧 CAD Chat Agent")
    print("="*60)
    print("\nBefehle:")
    print("  /load <pfad>  - STEP-Datei laden")
    print("  /reset        - Conversation zurücksetzen")
    print("  /quit         - Beenden")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Du: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Auf Wiedersehen!")
            break
        
        if not user_input:
            continue
        
        # Befehle
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd == "/quit" or cmd == "/exit":
                print("👋 Auf Wiedersehen!")
                break
            
            elif cmd == "/reset":
                agent.reset_conversation()
                print("🔄 Conversation zurückgesetzt.")
                continue
            
            elif cmd == "/load":
                if len(parts) < 2:
                    print("❌ Usage: /load <pfad>")
                    continue
                try:
                    agent.set_step_file(parts[1])
                except FileNotFoundError as e:
                    print(f"❌ {e}")
                continue
            
            else:
                print(f"❌ Unbekannter Befehl: {cmd}")
                continue
        
        # Chat
        if not agent.step_path:
            print("❌ Lade zuerst eine STEP-Datei mit /load <pfad>")
            continue
        
        print("\nAgent: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()


def single_query(agent: CADChatAgent, query: str) -> str:
    """Einzelne Anfrage"""
    return agent.chat(query)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAD Chat Agent")
    parser.add_argument("step_file", nargs="?", help="STEP-Datei zum Analysieren")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Interaktiver Modus")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Ollama Modell (default: {DEFAULT_MODEL})")
    parser.add_argument("--query", "-q", help="Einzelne Frage stellen")
    
    args = parser.parse_args()
    
    # Agent erstellen
    agent = CADChatAgent(model=args.model)
    
    # STEP-Datei laden falls angegeben
    if args.step_file:
        try:
            agent.set_step_file(args.step_file)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
    
    # Modus wählen
    if args.query:
        # Einzelne Frage
        if not agent.step_path:
            print("❌ Keine STEP-Datei angegeben!")
            sys.exit(1)
        response = single_query(agent, args.query)
        print(response)
    
    elif args.interactive or not args.step_file:
        # Interaktiv
        interactive_mode(agent)
    
    else:
        # Default: Zusammenfassung ausgeben
        print("\n📊 Analysiere STEP-Datei...\n")
        response = agent.chat("Gib mir eine Zusammenfassung dieses Bauteils.")
        print(response)


if __name__ == "__main__":
    main()
