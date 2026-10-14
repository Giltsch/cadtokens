"""
Ballooned inspection report (PDF) from STEP AP242 semantic PMI.

Page 1 shows an isometric wireframe of the part with numbered balloons
pointing at each characteristic's feature; the following pages hold the
characteristic table with empty "Measured" and "OK" columns for the
inspector (AS9102 Form 3 / VDA 2 EMPB style).

Zero dependencies: the PDF is written directly, GD&T symbols are drawn as
vector graphics so they print correctly without special fonts.

    from cadtokens.report import write_report
    write_report("part.stp", "part_inspection.pdf")
"""

import datetime
import math
import os
import zlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cadtokens
from cadtokens.characteristics import (
    GDT_SYMBOLS, Characteristic, _num, build_characteristics,
)
from cadtokens.step_engine.model import StepFile
from cadtokens.step_engine.pmi import PMIData
from cadtokens.step_engine.wireframe import Wireframe

PAGE_W, PAGE_H = 842.0, 595.0          # A4 landscape, points
MARGIN = 28.0

BLUE = (0.10, 0.30, 0.70)
GREY = (0.55, 0.57, 0.60)
LIGHT = (0.90, 0.91, 0.93)
DARK = (0.12, 0.12, 0.14)
RED = (0.75, 0.12, 0.10)
AMBER = (1.00, 0.95, 0.75)

# Helvetica advance widths (1/1000 em) for ASCII 32–126
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]
_HELV_EXTRA = {'±': 584, '°': 400, '×': 584, 'µ': 556, '–': 556, '—': 1000}


def text_width(s: str, size: float, bold: bool = False) -> float:
    w = 0
    for ch in s:
        o = ord(ch)
        w += _HELV[o - 32] if 32 <= o <= 126 else _HELV_EXTRA.get(ch, 556)
    return w * size / 1000.0 * (1.05 if bold else 1.0)


def _pdf_string(s: str) -> str:
    raw = s.encode('cp1252', errors='replace').decode('latin-1')
    return '(' + raw.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)') + ')'


# ======================================================================
# Minimal PDF canvas
# ======================================================================

class Canvas:
    """Vector drawing operations for one PDF page (origin bottom-left)."""

    def __init__(self):
        self.ops: List[str] = []

    def _o(self, s: str):
        self.ops.append(s)

    # state
    def stroke(self, rgb): self._o('%.3f %.3f %.3f RG' % rgb)
    def fill(self, rgb): self._o('%.3f %.3f %.3f rg' % rgb)
    def width(self, w): self._o('%.2f w' % w)
    def dash(self, on=None, off=None): self._o('[%s] 0 d' % ('' if on is None else '%.1f %.1f' % (on, off)))
    def save(self): self._o('q')
    def restore(self): self._o('Q')

    # paths
    def line(self, x1, y1, x2, y2):
        self._o('%.2f %.2f m %.2f %.2f l S' % (x1, y1, x2, y2))

    def polyline(self, pts: Sequence[Tuple[float, float]], close=False, fill=False):
        if len(pts) < 2:
            return
        path = ['%.2f %.2f m' % pts[0]] + ['%.2f %.2f l' % p for p in pts[1:]]
        self._o(' '.join(path) + (' h' if close else '') + (' b' if fill else (' s' if close else ' S')))

    def rect(self, x, y, w, h, stroke=True, fill=False):
        op = 'B' if stroke and fill else ('f' if fill else 'S')
        self._o('%.2f %.2f %.2f %.2f re %s' % (x, y, w, h, op))

    def circle(self, cx, cy, r, stroke=True, fill=False):
        k = 0.5523 * r
        self._o(' '.join([
            '%.2f %.2f m' % (cx + r, cy),
            '%.2f %.2f %.2f %.2f %.2f %.2f c' % (cx + r, cy + k, cx + k, cy + r, cx, cy + r),
            '%.2f %.2f %.2f %.2f %.2f %.2f c' % (cx - k, cy + r, cx - r, cy + k, cx - r, cy),
            '%.2f %.2f %.2f %.2f %.2f %.2f c' % (cx - r, cy - k, cx - k, cy - r, cx, cy - r),
            '%.2f %.2f %.2f %.2f %.2f %.2f c' % (cx + k, cy - r, cx + r, cy - k, cx + r, cy),
            'B' if stroke and fill else ('f' if fill else 'S')]))

    def arc(self, cx, cy, r, a0, a1, close=False):
        """Circular arc from angle a0 to a1 (radians), as Bézier segments."""
        n = max(1, int(math.ceil(abs(a1 - a0) / (math.pi / 2))))
        step = (a1 - a0) / n
        k = 4.0 / 3.0 * math.tan(step / 4)
        ops = ['%.2f %.2f m' % (cx + r * math.cos(a0), cy + r * math.sin(a0))]
        for i in range(n):
            t0, t1 = a0 + i * step, a0 + (i + 1) * step
            p0 = (cx + r * math.cos(t0), cy + r * math.sin(t0))
            p3 = (cx + r * math.cos(t1), cy + r * math.sin(t1))
            p1 = (p0[0] - k * r * math.sin(t0), p0[1] + k * r * math.cos(t0))
            p2 = (p3[0] + k * r * math.sin(t1), p3[1] - k * r * math.cos(t1))
            ops.append('%.2f %.2f %.2f %.2f %.2f %.2f c' % (p1 + p2 + p3))
        self._o(' '.join(ops) + (' h S' if close else ' S'))

    def text(self, x, y, s, size=8.0, bold=False, align='left'):
        if align != 'left':
            w = text_width(s, size, bold)
            x -= w if align == 'right' else w / 2
        self._o('BT /%s %.2f Tf %.2f %.2f Td %s Tj ET' % ('F2' if bold else 'F1', size, x, y, _pdf_string(s)))


def write_pdf(pages: List[Canvas], path: str, title: str = ''):
    objs: List[bytes] = []

    def add(body: Union[str, bytes]) -> int:
        objs.append(body.encode('latin-1') if isinstance(body, str) else body)
        return len(objs)

    catalog = add('')
    pages_id = add('')
    f1 = add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>')
    f2 = add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>')
    kids = []
    for page in pages:
        data = zlib.compress('\n'.join(page.ops).encode('latin-1'))
        content = add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(data) + data + b'\nendstream')
        kids.append(add('<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.0f %.0f] '
                        '/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> /Contents %d 0 R >>'
                        % (pages_id, PAGE_W, PAGE_H, f1, f2, content)))
    objs[catalog - 1] = ('<< /Type /Catalog /Pages %d 0 R >>' % pages_id).encode()
    objs[pages_id - 1] = ('<< /Type /Pages /Kids [%s] /Count %d >>'
                          % (' '.join('%d 0 R' % k for k in kids), len(kids))).encode()
    info = add('<< /Producer (cadtokens %s) /Title %s >>' % (cadtokens.__version__, _pdf_string(title)))

    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b'%d 0 obj\n' % i + body + b'\nendobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objs) + 1)
    out += b''.join(b'%010d 00000 n \n' % o for o in offsets)
    out += b'trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (
        len(objs) + 1, catalog, info, xref)
    with open(path, 'wb') as f:
        f.write(bytes(out))


# ======================================================================
# GD&T symbols as vector graphics
# ======================================================================

_SYMBOL_NAMES = {v: k for k, v in GDT_SYMBOLS.items()}
_CIRCLED = {'Ⓜ': 'M', 'Ⓛ': 'L', 'Ⓕ': 'F', 'Ⓣ': 'T', 'Ⓟ': 'P', 'Ⓤ': 'U', 'Ⓡ': 'R'}


def draw_gdt_symbol(c: Canvas, name: str, x: float, y: float, s: float):
    """Draw a geometric characteristic symbol inside the square (x, y, s, s)."""
    cx, cy, r = x + s / 2, y + s / 2, s * 0.32
    L = c.line
    if name == 'POSITION':
        c.circle(cx, cy, r * 0.8)
        L(cx - r * 1.2, cy, cx + r * 1.2, cy)
        L(cx, cy - r * 1.2, cx, cy + r * 1.2)
    elif name == 'FLATNESS':
        c.polyline([(x + .12 * s, y + .3 * s), (x + .66 * s, y + .3 * s),
                    (x + .88 * s, y + .7 * s), (x + .34 * s, y + .7 * s)], close=True)
    elif name == 'STRAIGHTNESS':
        L(x + .12 * s, cy, x + .88 * s, cy)
    elif name == 'ROUNDNESS':
        c.circle(cx, cy, r)
    elif name == 'CYLINDRICITY':
        c.circle(cx, cy, r * 0.75)
        L(x + .12 * s, y + .12 * s, x + .48 * s, y + .9 * s)
        L(x + .52 * s, y + .1 * s, x + .88 * s, y + .88 * s)
    elif name in ('LINE_PROFILE', 'SURFACE_PROFILE'):
        c.arc(cx, y + .25 * s, s * .36, 0, math.pi, close=(name == 'SURFACE_PROFILE'))
    elif name == 'PERPENDICULARITY':
        L(x + .12 * s, y + .15 * s, x + .88 * s, y + .15 * s)
        L(cx, y + .15 * s, cx, y + .88 * s)
    elif name == 'PARALLELISM':
        L(x + .15 * s, y + .15 * s, x + .5 * s, y + .88 * s)
        L(x + .5 * s, y + .15 * s, x + .85 * s, y + .88 * s)
    elif name == 'ANGULARITY':
        L(x + .12 * s, y + .15 * s, x + .88 * s, y + .15 * s)
        L(x + .12 * s, y + .15 * s, x + .78 * s, y + .82 * s)
    elif name in ('CONCENTRICITY', 'COAXIALITY'):
        c.circle(cx, cy, r)
        c.circle(cx, cy, r * 0.5)
    elif name == 'SYMMETRY':
        L(x + .28 * s, y + .3 * s, x + .72 * s, y + .3 * s)
        L(x + .1 * s, cy, x + .9 * s, cy)
        L(x + .28 * s, y + .7 * s, x + .72 * s, y + .7 * s)
    elif name in ('CIRCULAR_RUNOUT', 'TOTAL_RUNOUT'):
        arrows = [(.2, .5)] if name == 'CIRCULAR_RUNOUT' else [(.12, .42), (.45, .75)]
        for a, b in arrows:
            x0, y0, x1, y1 = x + a * s, y + .15 * s, x + (b + .25) * s, y + .85 * s
            L(x0, y0, x1, y1)
            ang = math.atan2(y1 - y0, x1 - x0)
            h = s * 0.22
            c.polyline([(x1, y1),
                        (x1 - h * math.cos(ang - .4), y1 - h * math.sin(ang - .4)),
                        (x1 - h * math.cos(ang + .4), y1 - h * math.sin(ang + .4))],
                       close=True, fill=True)
        if name == 'TOTAL_RUNOUT':
            L(x + .1 * s, y + .15 * s, x + .9 * s, y + .15 * s)
    else:
        c.text(cx, y + .25 * s, name[:2], s * .6, align='center')


def _tokens(text: str) -> List[Tuple]:
    """Split a specification into text and symbol tokens."""
    out: List[Tuple] = []
    buf = ''
    i = 0
    while i < len(text):
        ch = text[i]
        special = None
        if ch == '⌀':
            special = ('diam',)
        elif ch in _CIRCLED:
            special = ('circ', _CIRCLED[ch])
        elif ch in _SYMBOL_NAMES:
            special = ('gdt', _SYMBOL_NAMES[ch])
        elif text.startswith('⟨ST⟩', i):
            special = ('hex', 'ST')
            i += 3
        if special:
            if buf:
                out.append(('t', buf))
                buf = ''
            out.append(special)
        else:
            buf += ch
        i += 1
    if buf:
        out.append(('t', buf))
    return out


def rich_width(text: str, size: float) -> float:
    w = 0.0
    for tok in _tokens(text):
        if tok[0] == 't':
            w += text_width(tok[1], size)
        elif tok[0] in ('diam', 'gdt'):
            w += size * 0.95
        else:
            w += size * 1.05 + (size * 0.6 if tok[0] == 'hex' else 0)
    return w


def draw_rich(c: Canvas, x: float, y: float, text: str, size: float) -> float:
    """Draw text with GD&T glyphs at baseline y; returns the end x."""
    for tok in _tokens(text):
        if tok[0] == 't':
            c.text(x, y, tok[1], size)
            x += text_width(tok[1], size)
        elif tok[0] == 'diam':
            r = size * 0.3
            cx, cy = x + size * 0.45, y + size * 0.34
            c.circle(cx, cy, r)
            c.line(cx - r * 1.1, cy - r * 1.3, cx + r * 1.1, cy + r * 1.3)
            x += size * 0.95
        elif tok[0] == 'gdt':
            draw_gdt_symbol(c, tok[1], x, y - size * 0.15, size * 0.95)
            x += size * 0.95
        elif tok[0] == 'circ':
            r = size * 0.45
            c.circle(x + r + size * 0.05, y + size * 0.33, r)
            c.text(x + r + size * 0.05, y + size * 0.1, tok[1], size * 0.6, align='center')
            x += size * 1.05
        elif tok[0] == 'hex':
            w, h = size * 1.6, size * 0.95
            yb = y - size * 0.15
            c.polyline([(x, yb + h / 2), (x + .25 * h, yb), (x + w - .25 * h, yb),
                        (x + w, yb + h / 2), (x + w - .25 * h, yb + h), (x + .25 * h, yb + h)], close=True)
            c.text(x + w / 2, y, tok[1], size * 0.6, align='center')
            x += w + size * 0.05
    return x


def spec_width(char: Characteristic, size: float) -> float:
    if char.kind != 'geometric_tolerance':
        return rich_width(char.specification, size)
    prefix, cells = _frame_cells(char.specification)
    pad = size * 0.45
    return (text_width(prefix, size) if prefix else 0) + size * 1.4 + \
        sum(rich_width(cell, size) + 2 * pad for cell in cells)


def _frame_cells(spec: str) -> Tuple[str, List[str]]:
    parts = spec.split(' | ')
    first = parts[0]
    prefix = first.rsplit(' ', 1)[0] + ' ' if ' ' in first else ''
    return prefix, parts[1:]


def draw_spec(c: Canvas, x: float, y: float, char: Characteristic, size: float):
    """Specification at baseline y: a feature control frame for GD&T, text otherwise."""
    c.stroke(DARK)
    c.fill(DARK)
    c.width(0.5)
    if char.kind != 'geometric_tolerance':
        draw_rich(c, x, y, char.specification, size)
        return
    prefix, cells = _frame_cells(char.specification)
    symbol = char.specification.split(' | ')[0].strip().split(' ')[-1]
    if prefix:
        c.text(x, y, prefix, size)
        x += text_width(prefix, size)
    h = size * 1.4
    yb = y - size * 0.35
    c.rect(x, yb, h, h)
    draw_gdt_symbol(c, _SYMBOL_NAMES.get(symbol, symbol), x + h * .12, yb + h * .12, h * .76)
    x += h
    pad = size * 0.45
    for cell in cells:
        w = rich_width(cell, size) + 2 * pad
        c.rect(x, yb, w, h)
        draw_rich(c, x + pad, y, cell, size)
        x += w


# ======================================================================
# Balloon layout
# ======================================================================

def _ring_angles(ideal: List[float], min_gap: float) -> List[float]:
    """Spread angles around a ring: keep their order, at least min_gap apart,
    moving each as little as possible (cluster merging, cut at the widest gap)."""
    n = len(ideal)
    period = 2 * math.pi
    if n <= 1:
        return list(ideal)
    order = sorted(range(n), key=lambda i: ideal[i] % period)
    th = [ideal[i] % period for i in order]
    if n * min_gap >= period:
        return _unsort(order, [th[0] + period * k / n for k in range(n)])

    # Unwrap starting after the widest gap, so the cut does the least harm
    gaps = [th[k + 1] - th[k] for k in range(n - 1)] + [th[0] + period - th[-1]]
    s0 = (max(range(n), key=lambda k: gaps[k]) + 1) % n
    seq = [th[(s0 + k) % n] + (period if s0 + k >= n else 0.0) for k in range(n)]

    clusters = [[k] for k in range(n)]           # each: member indices into seq

    def first_last(cl):
        center = sum(seq[k] for k in cl) / len(cl)
        half = (len(cl) - 1) * min_gap / 2
        return center - half, center + half

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters) - 1):
            if first_last(clusters[i])[1] + min_gap > first_last(clusters[i + 1])[0] + 1e-12:
                clusters[i] += clusters.pop(i + 1)
                merged = True
                break
    placed = []
    for cl in clusters:
        lo, _ = first_last(cl)
        placed += [lo + j * min_gap for j in range(len(cl))]
    if placed[-1] - placed[0] > period - min_gap:          # wrapped onto itself
        placed = [placed[0] + period * k / n for k in range(n)]
    angles = [0.0] * n
    for k in range(n):
        angles[(k + s0) % n] = placed[k]
    return _unsort(order, angles)


def _unsort(order: List[int], values: List[float]) -> List[float]:
    result = [0.0] * len(order)
    for k, i in enumerate(order):
        result[i] = values[k]
    return result


def _up_axis_permutation(points) -> Tuple[int, int, int]:
    """Axis order that puts the part's thinnest extent vertical (parts lie flat)."""
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    up = min(range(3), key=lambda i: hi[i] - lo[i])
    return {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}[up]   # cyclic: keeps handedness


def _iso(p, az=math.radians(45), el=math.radians(35.264)):
    x, y, z = p
    u = x * math.cos(az) - y * math.sin(az)
    d = x * math.sin(az) + y * math.cos(az)
    return u, z * math.cos(el) + d * math.sin(el)


def draw_view(c: Canvas, wire: Wireframe, chars: List[Characteristic],
              box: Tuple[float, float, float, float]) -> int:
    """Wireframe + balloons inside box (x, y, w, h). Returns number of balloons drawn."""
    bx, by, bw, bh = box
    edges = wire.edges()
    if not edges:
        c.fill(GREY)
        c.text(bx + bw / 2, by + bh / 2, 'No B-Rep edges to display (tessellated or empty file).',
               10, align='center')
        return 0

    perm = _up_axis_permutation([p for pts in edges.values() for p in pts[::4] or pts])

    def view(p):
        return _iso((p[perm[0]], p[perm[1]], p[perm[2]]))

    proj = {eid: [view(p) for p in pts] for eid, pts in edges.items()}
    xs = [p[0] for pts in proj.values() for p in pts]
    ys = [p[1] for pts in proj.values() for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

    anchors: Dict[int, Tuple[float, float]] = {}
    for ch in chars:
        if ch.face_ids:
            a = wire.anchor(ch.face_ids)
            if a is not None:
                anchors[ch.number] = view(a)

    n = len(anchors)
    r = 7.0 if n <= 60 else 5.5
    rings = 1 if n * (2 * r + 3) < 2 * math.pi * min(bw, bh) * 0.42 else 2
    margin = (2 * r + 10) * rings + 6
    scale = min((bw - 2 * margin) / max(maxx - minx, 1e-9), (bh - 2 * margin) / max(maxy - miny, 1e-9))
    ox = bx + bw / 2 - (minx + maxx) / 2 * scale
    oy = by + bh / 2 - (miny + maxy) / 2 * scale

    def to_page(p):
        return ox + p[0] * scale, oy + p[1] * scale

    c.stroke(GREY)
    c.width(0.35)
    for pts in proj.values():
        c.polyline([to_page(p) for p in pts])

    if not n:
        return 0

    cx, cy = bx + bw / 2, by + bh / 2
    half_w, half_h = (maxx - minx) * scale / 2, (maxy - miny) * scale / 2
    numbers = sorted(anchors)
    pos = {k: to_page(anchors[k]) for k in numbers}

    ring_sets = [numbers] if rings == 1 else [numbers[0::2], numbers[1::2]]
    for ring_idx, members in enumerate(ring_sets):
        a = min(half_w + r + 8 + ring_idx * (2 * r + 6), bw / 2 - r - 2)
        b = min(half_h + r + 8 + ring_idx * (2 * r + 6), bh / 2 - r - 2)
        perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        min_gap = 2 * math.pi * (2 * r + 2.5) / perimeter
        ideal = [math.atan2((pos[k][1] - cy) / b, (pos[k][0] - cx) / a) for k in members]
        for k, ang in zip(members, _ring_angles(ideal, min_gap)):
            px, py = cx + a * math.cos(ang), cy + b * math.sin(ang)
            ax, ay = pos[k]
            dist = math.hypot(ax - px, ay - py) or 1.0
            c.stroke(BLUE)
            c.width(0.45)
            c.line(px + (ax - px) / dist * r, py + (ay - py) / dist * r, ax, ay)
            c.fill(BLUE)
            c.circle(ax, ay, 1.1, stroke=False, fill=True)
            c.fill((1, 1, 1))
            c.width(0.8)
            c.circle(px, py, r, stroke=True, fill=True)
            c.fill(BLUE)
            size = r * (1.05 if k < 10 else 0.95 if k < 100 else 0.78)
            c.text(px, py - size * 0.36, str(k), size, bold=True, align='center')
    return n


# ======================================================================
# Table
# ======================================================================

_COLUMNS = [  # (title, width)
    ('No.', 24), ('Characteristic', 80), ('Specification', 196), ('Nominal', 46),
    ('Min', 46), ('Max', 46), ('Unit', 24), ('Faces', 64), ('Measured', 62), ('OK', 24),
    ('Notes', 0),
]


def _wrap(text: str, width: float, size: float) -> List[str]:
    lines, line = [], ''
    for word in text.split():
        trial = (line + ' ' + word).strip()
        if text_width(trial, size) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _faces_text(ch: Characteristic) -> str:
    if ch.applies_to != 'faces':
        return ch.applies_to
    ids = ['#%d' % f for f in ch.face_ids[:3]]
    return ' '.join(ids) + (' +%d' % (len(ch.face_ids) - 3) if len(ch.face_ids) > 3 else '')


def _table_pages(chars: List[Characteristic], header: str) -> List[Canvas]:
    widths = [w for _, w in _COLUMNS]
    widths[-1] = PAGE_W - 2 * MARGIN - sum(widths)
    size, note_size = 7.0, 6.0
    pages: List[Canvas] = []
    c: Optional[Canvas] = None
    y = 0.0

    def new_page():
        nonlocal c, y
        c = Canvas()
        pages.append(c)
        c.fill(DARK)
        c.text(MARGIN, PAGE_H - MARGIN - 4, header, 10, bold=True)
        y = PAGE_H - MARGIN - 18
        c.fill(LIGHT)
        c.stroke(GREY)
        c.width(0.4)
        c.rect(MARGIN, y - 14, sum(widths), 14, stroke=True, fill=True)
        x = MARGIN
        c.fill(DARK)
        for (title, _), w in zip(_COLUMNS, widths):
            c.text(x + 3, y - 10, title, size, bold=True)
            x += w
        y -= 14

    new_page()
    for ch in chars:
        notes = _wrap(' '.join(ch.notes), widths[-1] - 6, note_size)
        row_h = max(15.0, len(notes) * (note_size + 1.5) + 5)
        if y - row_h < MARGIN + 16:
            new_page()
        attention = ch.lower_limit is None and ch.upper_limit is None
        if attention:
            c.fill(AMBER)
            c.rect(MARGIN, y - row_h, sum(widths), row_h, stroke=False, fill=True)
        c.stroke(GREY)
        c.width(0.3)
        c.rect(MARGIN, y - row_h, sum(widths), row_h)
        x = MARGIN
        for w in widths[:-1]:
            x += w
            c.line(x, y, x, y - row_h)

        base = y - 10
        dec = None
        cells = [str(ch.number), ch.characteristic, None,
                 '' if ch.nominal is None or ch.kind == 'geometric_tolerance' else _num(ch.nominal, dec),
                 '' if ch.lower_limit is None else _num(ch.lower_limit, dec),
                 '' if ch.upper_limit is None else _num(ch.upper_limit, dec),
                 ch.unit or '', _faces_text(ch), '', '']
        x = MARGIN
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i == 2:
                c.save()
                draw_spec(c, x + 3, base, ch, size)
                c.restore()
            elif cell:
                c.fill(BLUE if i == 0 else DARK)
                txt = cell
                while text_width(txt, size, i == 0) > w - 5 and len(txt) > 2:
                    txt = txt[:-2] + '…'
                c.text(x + 3, base, txt, size, bold=(i == 0))
            x += w
        c.fill(RED if attention else DARK)
        for k, line in enumerate(notes):
            c.text(x + 3, y - 8 - k * (note_size + 1.5), line, note_size)
        y -= row_h
    return pages


# ======================================================================
# Report
# ======================================================================

def write_report(step: Union[str, StepFile], output_path: str,
                 pmi: Optional[PMIData] = None,
                 chars: Optional[List[Characteristic]] = None,
                 title: Optional[str] = None) -> Dict:
    """Write a ballooned inspection report PDF.

    Args:
        step: path to a STEP file, or a parsed StepFile
        output_path: PDF file to write
        pmi / chars: reuse already extracted data
        title: report title (default: file name)
    Returns:
        {'pages', 'characteristics', 'ballooned', 'warnings'}
    """
    source = step if isinstance(step, str) else (step.header.file_name or 'STEP file')
    sf = cadtokens.load(step) if isinstance(step, str) else step
    graph = cadtokens.graph(sf)
    if pmi is None:
        pmi = cadtokens.extract_pmi(sf, graph)
    if chars is None:
        chars = build_characteristics(pmi)
    name = os.path.basename(source)
    title = title or name
    today = datetime.date.today().isoformat()

    # ---- page 1: view with balloons ---------------------------------
    p1 = Canvas()
    p1.fill(DARK)
    p1.text(MARGIN, PAGE_H - MARGIN - 6, 'Inspection Characteristic Report', 15, bold=True)
    p1.text(MARGIN, PAGE_H - MARGIN - 22, title, 10)
    info = [('Source', name), ('Date', today), ('Length unit', pmi.length_unit or '-'),
            ('Characteristics', str(len(chars)))]
    x = PAGE_W - MARGIN - 250
    for k, (label, value) in enumerate(info):
        p1.fill(GREY)
        p1.text(x, PAGE_H - MARGIN - 6 - k * 10, label, 7)
        p1.fill(DARK)
        p1.text(x + 70, PAGE_H - MARGIN - 6 - k * 10, value[:48], 7, bold=True)
    top = PAGE_H - MARGIN - 44

    if pmi.warnings:
        lines = []
        for w in pmi.warnings:
            lines += _wrap('! ' + w, PAGE_W - 2 * MARGIN - 16, 7)
        shown = lines[:5] + (['… %d more — see the JSON/CSV export' % (len(lines) - 5)] if len(lines) > 5 else [])
        h = 14 + 9 * len(shown)
        p1.fill((1.0, 0.94, 0.93))
        p1.stroke(RED)
        p1.width(0.6)
        p1.rect(MARGIN, top - h, PAGE_W - 2 * MARGIN, h, stroke=True, fill=True)
        p1.fill(RED)
        p1.text(MARGIN + 8, top - 11, 'Data-quality warnings in this file', 8, bold=True)
        for k, line in enumerate(shown):
            p1.text(MARGIN + 8, top - 21 - 9 * k, line, 7)
        top -= h + 8

    legend_h = 16
    view = (MARGIN, MARGIN + legend_h + 10, PAGE_W - 2 * MARGIN, top - MARGIN - legend_h - 10)
    p1.stroke(LIGHT)
    p1.width(0.5)
    p1.rect(*view)
    ballooned = draw_view(p1, Wireframe(graph), chars, view)
    unplaced = len(chars) - ballooned
    p1.fill(GREY)
    legend = ('Isometric wireframe. Balloon numbers refer to the characteristic table.'
              + (' %d characteristic(s) on the whole part or on supplemental geometry are listed in the'
                 ' table only.' % unplaced if unplaced else ''))
    p1.text(MARGIN, MARGIN + 14, legend, 7)

    # ---- table pages -------------------------------------------------
    pages = [p1] + _table_pages(chars, 'Characteristics — %s (%s)' % (name, pmi.length_unit or 'unit n/a'))

    for i, page in enumerate(pages, start=1):
        page.fill(GREY)
        page.text(MARGIN, MARGIN - 12,
                  'Generated by cadtokens %s from the semantic PMI in the STEP file. '
                  'Verify against the released model/drawing.' % cadtokens.__version__, 6.5)
        page.text(PAGE_W - MARGIN, MARGIN - 12, 'Page %d of %d' % (i, len(pages)), 6.5, align='right')

    write_pdf(pages, output_path, title='Inspection report - ' + name)
    return {'pages': len(pages), 'characteristics': len(chars), 'ballooned': ballooned,
            'warnings': list(pmi.warnings)}
