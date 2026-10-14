#!/usr/bin/env python3
"""
STEP Validator - Vergleicht GNN-Graph mit STEP-Rohtext.

Parst STEP-Entitäten direkt aus dem Text und vergleicht mit PythonOCC-Extraktion.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class STEPEntityCounts:
    """Zählung der STEP-Entitäten aus Rohtext."""
    advanced_faces: int = 0
    edge_curves: int = 0
    vertex_points: int = 0
    planes: int = 0
    cylindrical_surfaces: int = 0
    conical_surfaces: int = 0
    spherical_surfaces: int = 0
    toroidal_surfaces: int = 0
    b_spline_surfaces: int = 0
    circles: int = 0
    lines: int = 0
    b_spline_curves: int = 0
    

def parse_step_entities(step_path: str) -> STEPEntityCounts:
    """
    Parst STEP-Datei und zählt Entitäten direkt aus dem Rohtext.
    
    Args:
        step_path: Pfad zur STEP-Datei
        
    Returns:
        STEPEntityCounts mit Zählungen
    """
    counts = STEPEntityCounts()
    
    # Patterns für STEP-Entitäten
    patterns = {
        'advanced_faces': r'ADVANCED_FACE\s*\(',
        'edge_curves': r'EDGE_CURVE\s*\(',
        'vertex_points': r'VERTEX_POINT\s*\(',
        'planes': r'PLANE\s*\(',
        'cylindrical_surfaces': r'CYLINDRICAL_SURFACE\s*\(',
        'conical_surfaces': r'CONICAL_SURFACE\s*\(',
        'spherical_surfaces': r'SPHERICAL_SURFACE\s*\(',
        'toroidal_surfaces': r'TOROIDAL_SURFACE\s*\(',
        'b_spline_surfaces': r'B_SPLINE_SURFACE\s*\(',
        'circles': r'CIRCLE\s*\(',
        'lines': r'LINE\s*\(',
        'b_spline_curves': r'B_SPLINE_CURVE\s*\(',
    }
    
    with open(step_path, 'r', errors='ignore') as f:
        content = f.read()
    
    for attr, pattern in patterns.items():
        count = len(re.findall(pattern, content, re.IGNORECASE))
        setattr(counts, attr, count)
    
    return counts


def validate_gnn_against_step(step_path: str, gnn_data: Dict) -> Dict:
    """
    Validiert GNN-Graph gegen STEP-Rohtext.
    
    Args:
        step_path: Pfad zur STEP-Datei
        gnn_data: GNN-Daten aus PythonOCC
        
    Returns:
        Validierungsergebnis mit Vergleich und Warnings
    """
    # STEP-Rohtext parsen
    step_counts = parse_step_entities(step_path)
    
    # GNN-Daten extrahieren
    gnn_nodes = len(gnn_data.get('nodes', []))
    gnn_edges = len(gnn_data.get('edge_index', [[]])[0]) // 2 if gnn_data.get('edge_index') else 0
    
    # Surface-Typen aus GNN zählen
    gnn_surface_types = {}
    for node in gnn_data.get('nodes', []):
        stype = node.get('surface_type', 'unknown')
        gnn_surface_types[stype] = gnn_surface_types.get(stype, 0) + 1
    
    # Vergleich
    result = {
        'valid': True,
        'warnings': [],
        'comparison': {
            'faces': {
                'step_raw': step_counts.advanced_faces,
                'gnn': gnn_nodes,
                'match': step_counts.advanced_faces == gnn_nodes
            },
            'edges': {
                'step_raw': step_counts.edge_curves,
                'gnn_adjacencies': gnn_edges,
                # Edges im GNN sind Adjacencies, nicht 1:1 mit EDGE_CURVE
            },
            'vertices': {
                'step_raw': step_counts.vertex_points,
            },
            'surface_types': {
                'step_planes': step_counts.planes,
                'step_cylinders': step_counts.cylindrical_surfaces,
                'step_cones': step_counts.conical_surfaces,
                'step_spheres': step_counts.spherical_surfaces,
                'step_tori': step_counts.toroidal_surfaces,
                'step_bsplines': step_counts.b_spline_surfaces,
                'gnn_types': gnn_surface_types
            }
        }
    }
    
    # Validierung
    face_diff = abs(step_counts.advanced_faces - gnn_nodes)
    if face_diff > 0:
        result['warnings'].append(
            f"Face-Anzahl Differenz: STEP={step_counts.advanced_faces}, GNN={gnn_nodes} (Δ={face_diff})"
        )
        if face_diff > step_counts.advanced_faces * 0.1:  # >10% Abweichung
            result['valid'] = False
    
    # Surface-Type Validierung
    step_total_surfaces = (
        step_counts.planes + 
        step_counts.cylindrical_surfaces + 
        step_counts.conical_surfaces +
        step_counts.spherical_surfaces +
        step_counts.toroidal_surfaces +
        step_counts.b_spline_surfaces
    )
    
    if step_total_surfaces != step_counts.advanced_faces:
        result['warnings'].append(
            f"Surface-Summe ({step_total_surfaces}) ≠ Faces ({step_counts.advanced_faces}) - evtl. andere Surface-Typen"
        )
    
    return result


def validate_batch(entries: list, verbose: bool = True) -> Dict:
    """
    Validiert einen Batch von Dataset-Einträgen.
    
    Args:
        entries: Liste von Dataset-Einträgen
        verbose: Ausgabe pro Eintrag
        
    Returns:
        Zusammenfassung der Validierung
    """
    results = {
        'total': len(entries),
        'valid': 0,
        'invalid': 0,
        'warnings': 0,
        'details': []
    }
    
    for entry in entries:
        step_path = entry.get('step_path')
        gnn_data = entry.get('gnn_data', {})
        
        if not step_path or not Path(step_path).exists():
            results['invalid'] += 1
            continue
        
        validation = validate_gnn_against_step(step_path, gnn_data)
        
        if validation['valid']:
            results['valid'] += 1
        else:
            results['invalid'] += 1
            
        if validation['warnings']:
            results['warnings'] += len(validation['warnings'])
        
        if verbose and validation['warnings']:
            print(f"\n⚠️  {entry.get('id', 'unknown')}:")
            for w in validation['warnings']:
                print(f"   {w}")
        
        results['details'].append({
            'id': entry.get('id'),
            'validation': validation
        })
    
    return results


if __name__ == '__main__':
    import json
    import sys
    
    # Test mit Benchmark-Daten
    print("🔍 STEP Validator - Vergleich GNN vs. Rohtext")
    print("=" * 60)
    
    # Lade Benchmark
    entries = []
    with open('data/benchmark_output/benchmark_10_parts.json') as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"\n📊 Validiere {len(entries)} Einträge...\n")
    
    results = validate_batch(entries, verbose=True)
    
    print("\n" + "=" * 60)
    print("📋 ZUSAMMENFASSUNG:")
    print(f"   ✅ Valid:    {results['valid']}/{results['total']}")
    print(f"   ❌ Invalid:  {results['invalid']}/{results['total']}")
    print(f"   ⚠️  Warnings: {results['warnings']}")
    
    # Detaillierter Vergleich für erstes Teil
    if entries:
        print("\n" + "=" * 60)
        print("🔬 DETAILVERGLEICH (erstes Teil):")
        entry = entries[0]
        step_counts = parse_step_entities(entry['step_path'])
        
        print(f"\n   STEP-Rohtext ({Path(entry['step_path']).name}):")
        print(f"      ADVANCED_FACE:      {step_counts.advanced_faces}")
        print(f"      EDGE_CURVE:         {step_counts.edge_curves}")
        print(f"      VERTEX_POINT:       {step_counts.vertex_points}")
        print(f"      PLANE:              {step_counts.planes}")
        print(f"      CYLINDRICAL_SURFACE:{step_counts.cylindrical_surfaces}")
        print(f"      CONICAL_SURFACE:    {step_counts.conical_surfaces}")
        print(f"      SPHERICAL_SURFACE:  {step_counts.spherical_surfaces}")
        print(f"      B_SPLINE_SURFACE:   {step_counts.b_spline_surfaces}")
        
        gnn = entry.get('gnn_data', {})
        print(f"\n   GNN-Graph (PythonOCC):")
        print(f"      Nodes (Faces):      {len(gnn.get('nodes', []))}")
        print(f"      Edges (Adjacencies):{len(gnn.get('edge_index', [[]])[0]) // 2 if gnn.get('edge_index') else 0}")
        
        # Surface-Typen aus GNN
        gnn_types = {}
        for node in gnn.get('nodes', []):
            t = node.get('surface_type', 'unknown')
            gnn_types[t] = gnn_types.get(t, 0) + 1
        print(f"      Surface-Typen:      {gnn_types}")
