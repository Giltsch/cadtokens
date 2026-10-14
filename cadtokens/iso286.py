"""
ISO 286-1 limits and fits: deviations for tolerance classes such as H7, g6, G6.

Covers nominal sizes up to 500 mm, IT grades 1–18 and the fundamental
deviations used in practice:

    holes:  C D E F G H JS K M N P R S
    shafts: c d e f g h js k m n p r s

Anything outside that range raises UnsupportedFit — limits are never guessed.
Values are the tabulated ISO 286-1 values in micrometres.

Usage:
    from cadtokens.iso286 import fit_deviations
    fit_deviations(25.0, "H7")   # → (0.0, 0.021)   lower, upper in mm
    fit_deviations(25.0, "g6")   # → (-0.020, -0.007)
"""

import re
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple


class UnsupportedFit(ValueError):
    """The tolerance class or nominal size is not covered by the tables."""


# Upper bounds of the main size ranges in mm: "above previous, up to and including"
_MAIN = [3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400, 500]

# Upper bounds of the intermediate ranges (used by c, r, s)
_SUB = [3, 6, 10, 18, 30, 40, 50, 65, 80, 100, 120, 140, 160, 180,
        200, 225, 250, 280, 315, 355, 400, 450, 500]

# Standard tolerance grades IT1–IT18 in µm per main range (ISO 286-1, Table 1)
IT_GRADES: Dict[int, List[float]] = {
    1:  [0.8, 1, 1, 1.2, 1.5, 1.5, 2, 2.5, 3.5, 4.5, 6, 7, 8],
    2:  [1.2, 1.5, 1.5, 2, 2.5, 2.5, 3, 4, 5, 7, 8, 9, 10],
    3:  [2, 2.5, 2.5, 3, 4, 4, 5, 6, 8, 10, 12, 13, 15],
    4:  [3, 4, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
    5:  [4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27],
    6:  [6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40],
    7:  [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63],
    8:  [14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97],
    9:  [25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155],
    10: [40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250],
    11: [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400],
    12: [100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570, 630],
    13: [140, 180, 220, 270, 330, 390, 460, 540, 630, 720, 810, 890, 970],
    14: [250, 300, 360, 430, 520, 620, 740, 870, 1000, 1150, 1300, 1400, 1550],
    15: [400, 480, 580, 700, 840, 1000, 1200, 1400, 1600, 1850, 2100, 2300, 2500],
    16: [600, 750, 900, 1100, 1300, 1600, 1900, 2200, 2500, 2900, 3200, 3600, 4000],
    17: [1000, 1200, 1500, 1800, 2100, 2500, 3000, 3500, 4000, 4600, 5200, 5700, 6300],
    18: [1400, 1800, 2200, 2700, 3300, 3900, 4600, 5400, 6300, 7200, 8100, 8900, 9700],
}

# Shaft upper deviation es (µm), main ranges (ISO 286-1, Table 2)
_SHAFT_ES_MAIN = {
    'd': [-20, -30, -40, -50, -65, -80, -100, -120, -145, -170, -190, -210, -230],
    'e': [-14, -20, -25, -32, -40, -50, -60, -72, -85, -100, -110, -125, -135],
    'f': [-6, -10, -13, -16, -20, -25, -30, -36, -43, -50, -56, -62, -68],
    'g': [-2, -4, -5, -6, -7, -9, -10, -12, -14, -15, -17, -18, -20],
    'h': [0] * 13,
}
# Shaft upper deviation es (µm), intermediate ranges
_SHAFT_ES_SUB = {
    'c': [-60, -70, -80, -95, -110, -120, -130, -140, -150, -170, -180, -200, -210,
          -230, -240, -260, -280, -300, -330, -360, -400, -440, -480],
}
# Shaft lower deviation ei (µm), main ranges
_SHAFT_EI_MAIN = {
    'm': [2, 4, 6, 7, 8, 9, 11, 13, 15, 17, 20, 21, 23],
    'n': [4, 8, 10, 12, 15, 17, 20, 23, 27, 31, 34, 37, 40],
    'p': [6, 12, 15, 18, 22, 26, 32, 37, 43, 50, 56, 62, 68],
}
# k: ei for grades IT4–IT7; 0 for IT3 and below and IT8 and above
_SHAFT_K_EI = [0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5]
# Shaft lower deviation ei (µm), intermediate ranges
_SHAFT_EI_SUB = {
    'r': [10, 15, 19, 23, 28, 34, 34, 41, 43, 51, 54, 63, 65, 68,
          77, 80, 84, 94, 98, 108, 114, 126, 132],
    's': [14, 19, 23, 28, 35, 43, 43, 53, 59, 71, 79, 92, 100, 108,
          122, 130, 140, 158, 170, 190, 208, 232, 252],
}

_SHAFT_LETTERS = {'c', 'd', 'e', 'f', 'g', 'h', 'js', 'k', 'm', 'n', 'p', 'r', 's'}

_DESIGNATION = re.compile(r'^\s*([A-Za-z]{1,2})\s*(\d{1,2})\s*$')


def _range_index(nominal_mm: float, bounds: List[float]) -> int:
    if not 0 < nominal_mm <= bounds[-1]:
        raise UnsupportedFit(f"Nominal size {nominal_mm} mm outside 0–{bounds[-1]} mm")
    return bisect_left(bounds, nominal_mm)


def it_value(nominal_mm: float, grade: int) -> float:
    """Standard tolerance ITgrade in µm for a nominal size."""
    if grade not in IT_GRADES:
        raise UnsupportedFit(f"IT{grade} not covered (IT1–IT18)")
    return float(IT_GRADES[grade][_range_index(nominal_mm, _MAIN)])


def _delta(nominal_mm: float, grade: int) -> float:
    """Δ = ITn − IT(n−1), used for holes K…S in fine grades (ISO 286-1, Table 3)."""
    if nominal_mm <= 3 or grade < 3 or grade > 8:
        return 0.0
    return it_value(nominal_mm, grade) - it_value(nominal_mm, grade - 1)


def _shaft_fundamental(nominal_mm: float, letter: str, grade: int) -> Tuple[str, float]:
    """('es' or 'ei', value in µm) for a shaft letter."""
    if letter in _SHAFT_ES_MAIN:
        return 'es', float(_SHAFT_ES_MAIN[letter][_range_index(nominal_mm, _MAIN)])
    if letter in _SHAFT_ES_SUB:
        return 'es', float(_SHAFT_ES_SUB[letter][_range_index(nominal_mm, _SUB)])
    if letter in _SHAFT_EI_MAIN:
        return 'ei', float(_SHAFT_EI_MAIN[letter][_range_index(nominal_mm, _MAIN)])
    if letter in _SHAFT_EI_SUB:
        return 'ei', float(_SHAFT_EI_SUB[letter][_range_index(nominal_mm, _SUB)])
    if letter == 'k':
        ei = _SHAFT_K_EI[_range_index(nominal_mm, _MAIN)] if 4 <= grade <= 7 else 0
        return 'ei', float(ei)
    raise UnsupportedFit(f"Fundamental deviation '{letter}' not covered")


def fit_deviations_um(nominal_mm: float, designation: str,
                      kind: Optional[str] = None) -> Tuple[float, float]:
    """(lower, upper) deviation in µm for e.g. 'H7' (hole) or 'g6' (shaft).

    Args:
        nominal_mm: nominal size in mm
        designation: tolerance class, e.g. 'H7', 'g6', 'JS7'
        kind: 'hole' or 'shaft'; if omitted, upper case = hole, lower case = shaft
    """
    m = _DESIGNATION.match(designation or '')
    if not m:
        raise UnsupportedFit(f"Not an ISO 286 tolerance class: {designation!r}")
    letters, grade = m.group(1), int(m.group(2))
    if kind is None:
        kind = 'hole' if letters[0].isupper() else 'shaft'
    if kind not in ('hole', 'shaft'):
        raise UnsupportedFit(f"Unknown fit kind {kind!r}")
    letter = letters.lower()
    if letter not in _SHAFT_LETTERS:
        raise UnsupportedFit(f"Fundamental deviation '{letters}' not covered")

    it = it_value(nominal_mm, grade)

    if letter == 'js':
        return -it / 2, it / 2

    side, value = _shaft_fundamental(nominal_mm, letter, grade)

    if kind == 'shaft':
        return (value - it, value) if side == 'es' else (value, value + it)

    # Holes: A–H mirror the shaft (EI = −es)
    if side == 'es':
        ei_hole = -value
        return ei_hole, ei_hole + it

    # Holes K–S: ES = −ei, plus Δ in the fine grades (ISO 286-1, 4.2.3)
    if letter in ('k', 'm', 'n'):
        if grade > 8:
            raise UnsupportedFit(f"{letters.upper()}{grade}: grades above 8 not covered")
        es_hole = -value + _delta(nominal_mm, grade)
    else:  # p, r, s
        es_hole = -value + (_delta(nominal_mm, grade) if grade <= 7 else 0.0)
    return es_hole - it, es_hole


def fit_deviations(nominal_mm: float, designation: str,
                   kind: Optional[str] = None) -> Tuple[float, float]:
    """(lower, upper) deviation in mm, e.g. fit_deviations(25, 'H7') → (0.0, 0.021)."""
    lo, up = fit_deviations_um(nominal_mm, designation, kind)
    return round(lo / 1000.0, 6), round(up / 1000.0, 6)
