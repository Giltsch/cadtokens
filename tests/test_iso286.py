"""ISO 286 limits and fits."""

import math

import pytest

from cadtokens.iso286 import (
    IT_GRADES, UnsupportedFit, _MAIN, fit_deviations, fit_deviations_um, it_value,
)

# Values as printed in ISO 286-2 tables (µm): (nominal mm, class, lower, upper)
PUBLISHED = [
    (25, "H7", 0, 21), (25, "H8", 0, 33), (25, "H6", 0, 13), (100, "H11", 0, 220),
    (25, "h6", -13, 0), (25, "h7", -21, 0), (25, "g6", -20, -7), (25, "f7", -41, -20),
    (50, "f7", -50, -25), (25, "e8", -73, -40), (25, "d9", -117, -65),
    (25, "c11", -240, -110), (25, "k6", 2, 15), (25, "m6", 8, 21), (25, "n6", 15, 28),
    (25, "p6", 22, 35), (25, "r6", 28, 41), (25, "s6", 35, 48),
    (25, "js6", -6.5, 6.5), (25, "JS7", -10.5, 10.5),
    (25, "G7", 7, 28), (25, "F8", 20, 53), (25, "E9", 40, 92), (25, "D10", 65, 149),
    (25, "K7", -15, 6), (25, "K6", -11, 2), (25, "M7", -21, 0), (25, "M6", -17, -4),
    (25, "N7", -28, -7), (25, "N6", -24, -11), (25, "P7", -35, -14), (25, "P6", -31, -18),
    (25, "R7", -41, -20), (25, "S7", -48, -27),
    (3, "K7", -10, 0), (3, "M7", -12, -2), (3, "N7", -14, -4),
    (10, "H7", 0, 15), (10.0001, "H7", 0, 18),           # range boundary: up to and including
    (3.5, "G6", 4, 12),                                   # NIST FTC-10 holes
    (60, "s6", 53, 72), (70, "s6", 59, 78), (60, "r6", 41, 60),
]


@pytest.mark.parametrize("nominal,cls,lower,upper", PUBLISHED)
def test_published_values(nominal, cls, lower, upper):
    assert fit_deviations_um(nominal, cls) == pytest.approx((lower, upper))


def test_mm_conversion():
    assert fit_deviations(25, "H7") == (0.0, 0.021)
    assert fit_deviations(25, "g6") == (-0.020, -0.007)


def test_kind_overrides_case():
    assert fit_deviations_um(25, "G7", kind="shaft") == fit_deviations_um(25, "g7")


@pytest.mark.parametrize("grade,factor", [(5, 7), (6, 10), (7, 16), (8, 25), (9, 40), (10, 64),
                                          (11, 100), (12, 160), (13, 250), (14, 400),
                                          (15, 640), (16, 1000), (17, 1600), (18, 2500)])
def test_it_table_matches_standard_formula(grade, factor):
    """IT5–IT18 = factor × i, i = 0.45·∛D + 0.001·D (D = geometric mean of the range).

    The tables are rounded, so allow a few percent — this catches typos, not
    rounding. The two smallest ranges (≤ 6 mm) are rounded up in the standard
    by a constant ~10–15 % in every grade, so they get a wider band.
    """
    lower = 1.0
    for upper, tabulated in zip(_MAIN, IT_GRADES[grade]):
        d = math.sqrt(lower * upper)
        i = 0.45 * d ** (1 / 3) + 0.001 * d
        rel = 0.16 if upper <= 6 else 0.06
        assert tabulated == pytest.approx(factor * i, rel=rel), (grade, upper)
        lower = upper


def test_it_grows_with_grade_and_size():
    for grade in range(1, 18):
        assert all(a < b for a, b in zip(IT_GRADES[grade], IT_GRADES[grade + 1]))
    for grade in range(1, 19):
        assert all(a <= b for a, b in zip(IT_GRADES[grade], IT_GRADES[grade][1:]))
    assert it_value(25, 7) == 21


@pytest.mark.parametrize("nominal,cls", [(600, "H7"), (0, "H7"), (25, "ZC7"), (25, "H19"),
                                         (25, "K9"), (25, "M3x0.5"), (25, "")])
def test_unsupported_is_refused(nominal, cls):
    with pytest.raises(UnsupportedFit):
        fit_deviations_um(nominal, cls)
