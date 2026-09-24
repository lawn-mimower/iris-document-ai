"""Answer normalisation and scoring for both benchmarks.

DAM field extraction
    score_field(field_type, gold, predicted) compares one answer with the ground truth after
    normalising it for its type (identifier, text, e-mail, date, amount, yes/no). A gold value of
    None means the documents do not state the field; the answer is then correct only if the
    method declines to answer.
    source_correct(predicted_source, gold_sources) checks a returned (file, page) citation.

Form filling
    score_form(plan, gold_fields) checks a fill plan ([{"text", "bbox"}, ...], PDF points,
    top-left origin) against the ground-truth boxes: a text value counts as placed correctly when
    an entry with that text has its centre inside the target box, a checkbox when the chosen
    option's box (and no other option of the same group) holds a tick.
"""

import re
from datetime import date

# --------------------------------------------------------------------------- DAM normalisers

_UNANSWERED = re.compile(
    r"^\s*$|unanswerable|not\s+found|not\s+available|not\s+(stated|mentioned|provided|given|specified|disclosed)|"
    r"^\s*(unknown|none|null|nil|-+)\s*\.?\s*$|cannot\s+be\s+determined|no\s+information",
    re.IGNORECASE,
)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
MONTHS.update({name[:3]: num for name, num in list(MONTHS.items())})
MONTHS["sept"] = 9

_MONTH_RX = r"(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"


def is_unanswered(value) -> bool:
    if value is None:
        return True
    return bool(_UNANSWERED.search(str(value)))


def _safe_date(y: int, m: int, d: int):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(value):
    """Parse the date formats seen in the documents and in model answers. Day-first (Indian usage)."""
    if value is None:
        return None
    s = str(value).strip().lower().replace(",", " ")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b", s)
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[ \-]" + _MONTH_RX + r"[ \-](\d{4})\b", s)
    if m:
        return _safe_date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b" + _MONTH_RX + r" (\d{1,2}) (\d{4})\b", s)
    if m:
        return _safe_date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    return None


_MULTIPLIERS = [
    (re.compile(r"^\s*(crores?|cr\b\.?)", re.I), 10_000_000),
    (re.compile(r"^\s*(lakhs?|lacs?|lakh)", re.I), 100_000),
    (re.compile(r"^\s*(millions?|mn\b)", re.I), 1_000_000),
    (re.compile(r"^\s*(thousands?)", re.I), 1_000),
]


def parse_amount(value):
    """Parse an amount in rupees; understands Indian and international digit grouping, lakh/crore."""
    if value is None:
        return None
    s = str(value)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None
    number = float(m.group(0).replace(",", ""))
    rest = s[m.end():]
    for rx, mult in _MULTIPLIERS:
        if rx.match(rest):
            number *= mult
            break
    return int(round(number))


def parse_yesno(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if re.match(r"^(yes|y|true)\b", s):
        return "yes"
    if re.match(r"^(no|n|false)\b", s) or re.match(r"^(not applicable|n/?a)\b", s):
        return "no"
    return None


_CIN = re.compile(r"[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}")


def normalise_id(value):
    """Upper-case alphanumerics; if a CIN-shaped identifier is embedded ("CIN: ..."), just that."""
    if value is None:
        return None
    s = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    m = _CIN.search(re.sub(r"\s", "", str(value).upper()))
    return m.group(0) if m else s


def normalise_email(value):
    if value is None:
        return None
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", str(value))
    return m.group(0).lower().rstrip(".") if m else None


_TEXT_SUBS = [
    (r"&", " and "), (r"\bpvt\b\.?", "private"), (r"\bltd\b\.?", "limited"), (r"\bco\b\.", "company"),
]


def text_tokens(value) -> list:
    if value is None:
        return []
    s = str(value).lower()
    for pat, rep in _TEXT_SUBS:
        s = re.sub(pat, rep, s)
    return re.findall(r"[a-z0-9]+", s)


def token_f1(gold, pred) -> float:
    g, p = text_tokens(gold), text_tokens(pred)
    if not g or not p:
        return 0.0
    common = 0
    remaining = list(g)
    for tok in p:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


TEXT_F1_THRESHOLD = 0.8


def normalise(field_type: str, value):
    """Canonical form of an answer for its field type (None if it cannot be read)."""
    if field_type != "yesno" and is_unanswered(value):
        return None
    if field_type == "id":
        return normalise_id(value) or None
    if field_type == "email":
        return normalise_email(value)
    if field_type == "date":
        d = parse_date(value)
        return d.isoformat() if d else None
    if field_type == "amount":
        return parse_amount(value)
    if field_type == "yesno":
        return parse_yesno(value)
    if field_type == "text":
        return " ".join(text_tokens(value)) or None
    raise ValueError(f"unknown field type {field_type!r}")


def score_field(field_type: str, gold, predicted) -> dict:
    """Score one answer. Returns {correct, strict, gold_norm, pred_norm, answered}.

    correct: normalised match; for text fields a token F1 of at least 0.8 also counts.
    strict:  normalised exact match (identical to correct for non-text fields).
    """
    answered = not is_unanswered(predicted) if field_type != "yesno" else parse_yesno(predicted) is not None
    pred_norm = normalise(field_type, predicted)
    if gold is None:
        ok = not answered
        return {"correct": ok, "strict": ok, "gold_norm": None, "pred_norm": pred_norm, "answered": answered}
    gold_norm = normalise(field_type, gold)
    strict = pred_norm is not None and pred_norm == gold_norm
    correct = strict
    if field_type == "text" and not strict and answered:
        correct = token_f1(gold, predicted) >= TEXT_F1_THRESHOLD
    return {"correct": correct, "strict": strict, "gold_norm": gold_norm, "pred_norm": pred_norm,
            "answered": answered}


def source_correct(predicted_source, gold_sources) -> bool:
    """A citation is correct when its file holds the fact and, for PDF sources, the page matches."""
    if not predicted_source or not gold_sources:
        return False
    pf, pp = predicted_source.get("file"), predicted_source.get("page")
    for g in gold_sources:
        if g["file"] != pf:
            continue
        if g.get("page") is None or g["page"] == pp:
            return True
    return False


# --------------------------------------------------------------------------- form scoring

TICKS = {"✓", "✔", "☑", "x", "X", "✗", "yes", "Yes", "true", "True", "checked", "[x]", "tick"}


def bbox_centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def centre_in_box(pred_bbox, gold_bbox, tol: float = 2.0) -> bool:
    cx, cy = bbox_centre(pred_bbox)
    return gold_bbox[0] - tol <= cx <= gold_bbox[2] + tol and gold_bbox[1] - tol <= cy <= gold_bbox[3] + tol


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _norm_value(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _alnum(s) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).casefold())


def _valid_entry(entry) -> bool:
    try:
        b = entry["bbox"]
        return entry.get("text") is not None and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
    except (TypeError, KeyError, AttributeError):
        return False


def score_form(plan, gold_fields, text_tol: float = 2.0, tick_tol: float = 2.0) -> dict:
    """Score a fill plan against one form's ground truth.

    gold_fields: [{"key", "kind": "text", "value", "bbox"} |
                  {"key", "kind": "comb", "value", "bbox"} |      (character cells; bbox spans them)
                  {"key", "kind": "checkbox", "value", "options": [{"value", "bbox"}, ...]}]
    A comb field is correct when one entry with the value, or one entry per cell whose characters
    read the value left to right, sits inside its box (separators such as "/" are ignored).
    """
    entries = [e for e in (plan or []) if _valid_entry(e)]
    results = []
    for f in gold_fields:
        if f["kind"] == "text":
            want = _norm_value(f["value"])
            cands = [e for e in entries if _norm_value(e["text"]) == want]
            ok = any(centre_in_box(e["bbox"], f["bbox"], text_tol) for e in cands)
            best_iou = max((iou(e["bbox"], f["bbox"]) for e in cands), default=0.0)
            results.append({"key": f["key"], "kind": "text", "correct": ok, "placed": bool(cands),
                            "iou_pct": int(round(best_iou * 100))})
        elif f["kind"] == "comb":
            want = _alnum(f["value"])
            whole = any(_alnum(e["text"]) == want and centre_in_box(e["bbox"], f["bbox"], text_tol) for e in entries)
            inside = sorted((e for e in entries if centre_in_box(e["bbox"], f["bbox"], text_tol)),
                            key=lambda e: bbox_centre(e["bbox"])[0])
            per_cell = "".join(_alnum(e["text"]) for e in inside) == want
            results.append({"key": f["key"], "kind": "comb", "correct": whole or per_cell,
                            "placed": bool(inside)})
        else:
            want = _norm_value(f["value"])
            ticks = [e for e in entries
                     if str(e["text"]).strip() in TICKS or _norm_value(e["text"]) == want]
            ticked = [o["value"] for o in f["options"]
                      if any(centre_in_box(t["bbox"], o["bbox"], tick_tol) for t in ticks)]
            ok = ticked == [f["value"]]
            results.append({"key": f["key"], "kind": "checkbox", "correct": ok, "placed": bool(ticked),
                            "ticked": ticked})
    n = len(results)
    correct = sum(r["correct"] for r in results)
    return {"n_fields": n, "n_correct": correct,
            "n_plan_entries": len(plan or []), "n_invalid_entries": len(plan or []) - len(entries),
            "fields": results}
