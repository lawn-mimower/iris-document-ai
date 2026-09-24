"""Conventional form-filling baseline: nearest-label geometry, no LLM and no golden sample.

fill(elements, new_data) takes form elements ({"type": TEXT | INPUT_FIELD | CHECKBOX, "text", "bbox"},
PDF points, top-left origin) and returns a fill plan in the agent's format: [{"text", "bbox"}, ...].

    1. Clean the candidates: drop boxes whose centre lies inside a text span (letters that the
       shape detector reported as boxes) or that contain text (ruled label cells); treat small
       squares as checkboxes.
    2. Match each data key to the most similar label text (word overlap).
    3. Checkbox group: if a text near the label (same row to the right, or just below it) equals the
       value, tick the checkbox closest to that option text on the same row.
    4. Otherwise pick the nearest free box to the right of the label on the same row or below it;
       each box is used once, closest pairs first.

vector_elements(pdf) is a conventional perception step for born-digital PDFs: text spans from
PyMuPDF plus rectangles from the page's vector drawings, with ruled tables split into cells and
adjacent character cells merged. It is an alternative to phase1/phase1.py + gestalt_processor.py.
"""

import re
from difflib import SequenceMatcher

CHECK_MAX = 12.0      # squares up to this side (points) count as checkboxes
TICK = "✓"


def _tokens(s) -> list:
    words = re.findall(r"[a-z0-9]+", str(s).lower())
    return [w for w in words if not re.fullmatch(r"\d{1,2}|[a-h]", w)]  # drop item numbers "1", "a"


def label_similarity(key: str, label: str) -> float:
    k, t = _tokens(key), _tokens(label)
    if not k or not t:
        return 0.0
    common = len(set(k) & set(t))
    dice = 2 * common / (len(set(k)) + len(set(t)))
    ratio = SequenceMatcher(None, " ".join(k), " ".join(t)).ratio()
    return 0.8 * dice + 0.2 * ratio


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _inside(pt, b, pad=0.5):
    return b[0] - pad <= pt[0] <= b[2] + pad and b[1] - pad <= pt[1] <= b[3] + pad


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def prepare(elements):
    """Split elements into (texts, boxes, checkboxes) after removing artefacts.

    Any element with text counts as text (phase1 relabels some text spans as fields). A box is
    dropped when its centre lies inside a text span (a letter reported as a shape) or when a text
    span's centre lies inside it (a label cell, or a frame around printed text).
    """
    texts = [e for e in elements if e.get("text") is not None and str(e["text"]).strip()]
    boxes, checks = [], []
    for e in elements:
        if e.get("type") not in ("INPUT_FIELD", "CHECKBOX") or (e.get("text") and str(e["text"]).strip()):
            continue
        b = e["bbox"]
        if any(_inside(_centre(b), t["bbox"]) or _inside(_centre(t["bbox"]), b, pad=0) for t in texts):
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:
            continue
        square = 0.75 <= w / h <= 1.33
        if e["type"] == "CHECKBOX" or (square and max(w, h) <= CHECK_MAX):
            checks.append(e)
        else:
            boxes.append(e)
    return texts, boxes, checks


def _row_overlap(a, b, tol=6.0) -> bool:
    return abs(_centre(a)[1] - _centre(b)[1]) <= tol


def _box_distance(label, box):
    """Distance from a label to a box to its right on the same row or below it; None otherwise."""
    lb, bb = label["bbox"], box["bbox"]
    if _row_overlap(lb, bb, tol=max(6.0, (bb[3] - bb[1]) / 2)) and bb[0] >= lb[2] - 2:
        return bb[0] - lb[2]
    below = bb[1] - lb[3]
    if -2 <= below <= 40 and (min(lb[2], bb[2]) - max(lb[0], bb[0]) > 0 or abs(bb[0] - lb[0]) <= 20):
        return below + 1.0     # prefer the same row on a tie
    return None


def _option_text(label, value, texts):
    want = _norm(value)
    cands = []
    lb = label["bbox"]
    for t in texts:
        if t is label or _norm(t["text"]) != want:
            continue
        tb = t["bbox"]
        dy = _centre(tb)[1] - _centre(lb)[1]
        same_row = abs(dy) <= 6 and tb[0] >= lb[0]
        below = 0 < tb[1] - lb[3] + 2 <= 40
        if same_row or below:
            cands.append((abs(dy) + abs(tb[0] - lb[2]) * 0.01, t))
    return min(cands, key=lambda c: c[0])[1] if cands else None


def _nearest_check(option, checks):
    ob = option["bbox"]
    best = None
    for c in checks:
        cb = c["bbox"]
        if not _row_overlap(ob, cb):
            continue
        gap = cb[0] - ob[2] if cb[0] >= ob[2] else (ob[0] - cb[2] if cb[2] <= ob[0] else 0.0)
        if gap > 60:
            continue
        # on a tie prefer the box before the text (the more common convention)
        score = gap + (0.1 if cb[0] >= ob[2] else 0.0)
        if best is None or score < best[0]:
            best = (score, c)
    return best[1] if best else None


def fill(elements, new_data: dict, min_similarity: float = 0.45) -> list:
    texts, boxes, checks = prepare(elements)
    plan, text_jobs = [], []
    for key, value in new_data.items():
        scored = sorted(((label_similarity(key, t["text"]), i) for i, t in enumerate(texts)), reverse=True)
        if not scored or scored[0][0] < min_similarity:
            continue
        label = texts[scored[0][1]]
        option = _option_text(label, value, texts)
        if option is not None:
            box = _nearest_check(option, checks)
            if box is not None:
                plan.append({"text": TICK, "bbox": list(box["bbox"])})
                continue
        text_jobs.append((key, value, label))

    pairs = []
    for j, (key, value, label) in enumerate(text_jobs):
        for b, box in enumerate(boxes):
            d = _box_distance(label, box)
            if d is not None:
                pairs.append((d, j, b))
    used_jobs, used_boxes = set(), set()
    for d, j, b in sorted(pairs):
        if j in used_jobs or b in used_boxes:
            continue
        used_jobs.add(j)
        used_boxes.add(b)
        plan.append({"text": str(text_jobs[j][1]), "bbox": list(boxes[b]["bbox"])})
    return plan


# --------------------------------------------------------------------------- vector perception

def _merge_cells(rects, gap=6.5):
    """Merge same-row rectangles that touch or nearly touch (character cells) into one box."""
    rects = sorted(rects, key=lambda r: (round(r[1]), r[0]))
    merged = []
    for r in rects:
        if merged:
            m = merged[-1]
            same_row = abs(m[1] - r[1]) <= 1.5 and abs(m[3] - r[3]) <= 1.5
            if same_row and -1 <= r[0] - m[2] <= gap:
                merged[-1] = [m[0], min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3])]
                continue
        merged.append(list(r))
    return merged


def _grid_cells(hlines, vlines, tol=1.0):
    """Cells of ruled tables: rectangles bounded by line segments on all four sides."""
    ys = sorted({round(y, 1) for y, _, _ in hlines})
    xs = sorted({round(x, 1) for x, _, _ in vlines})

    def has_h(y, x0, x1):
        return any(abs(y - hy) <= tol and hx0 <= x0 + tol and hx1 >= x1 - tol for hy, hx0, hx1 in hlines)

    def has_v(x, y0, y1):
        return any(abs(x - vx) <= tol and vy0 <= y0 + tol and vy1 >= y1 - tol for vx, vy0, vy1 in vlines)

    cells = []
    for yi in range(len(ys) - 1):
        y0, y1 = ys[yi], ys[yi + 1]
        start = None
        for xi, x in enumerate(xs):
            if not has_v(x, y0, y1):
                continue
            if start is not None and has_h(y0, start, x) and has_h(y1, start, x):
                cells.append([start, y0, x, y1])
            start = x
    return cells


def vector_elements(pdf_path, page_no: int = 0) -> list:
    import fitz

    out = []
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_no]
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        out.append({"type": "TEXT", "text": span["text"].strip(),
                                    "bbox": [round(v, 2) for v in span["bbox"]]})
        rects, hlines, vlines = [], [], []
        for d in page.get_drawings():
            for item in d["items"]:
                if item[0] == "re":
                    r = item[1]
                    rects.append([r.x0, r.y0, r.x1, r.y1])
                elif item[0] == "l":
                    p, q = item[1], item[2]
                    if abs(p.y - q.y) < 0.5 and abs(p.x - q.x) > 2:
                        hlines.append((p.y, min(p.x, q.x), max(p.x, q.x)))
                    elif abs(p.x - q.x) < 0.5 and abs(p.y - q.y) > 2:
                        vlines.append((p.x, min(p.y, q.y), max(p.y, q.y)))
        small = [r for r in rects if max(r[2] - r[0], r[3] - r[1]) <= CHECK_MAX]
        large = [r for r in rects if max(r[2] - r[0], r[3] - r[1]) > CHECK_MAX]
        for r in small:
            out.append({"type": "CHECKBOX", "text": None, "bbox": [round(v, 2) for v in r]})
        for r in _merge_cells(large) + _grid_cells(hlines, vlines):
            out.append({"type": "INPUT_FIELD", "text": None, "bbox": [round(v, 2) for v in r]})
    return out
