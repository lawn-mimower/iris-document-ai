"""Baseline C: regular expressions and keyword rules, no model of any kind.

Works on the plain-text units from plain_text.extract_units (PDF pages first, then Word, then
spreadsheet sheets) and returns, for each field, an answer string (or None) with the file and page
of the unit it came from. The rules were written against the phrasing the benchmark generator uses,
so on these documents they are, if anything, optimistic for a rule-based system.
"""

import re
from datetime import timedelta

from scoring import parse_date

CIN_RX = re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RX = re.compile(
    r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MON + r",?\s+\d{4}\b"
    r"|\b" + _MON + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
    re.IGNORECASE,
)
NUMBER_RX = re.compile(r"(Rs\.?|INR|₹)?\s*(\d[\d,]*(?:\.\d+)?)(\s*(?:crores?|lakhs?|lacs?|millions?))?", re.IGNORECASE)
UNIT_MULT = [("crore", 10_000_000), ("lakh", 100_000), ("lac", 100_000), ("million", 1_000_000)]


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _src(unit):
    return {"file": unit["file"], "page": unit["page"]}


def _dates(text):
    return [m.group(0) for m in DATE_RX.finditer(text)]


def _sentences(text):
    return [s for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", _flat(text)) if s]


def _unit_multiplier(unit_text: str) -> int:
    header = unit_text.lower()
    for word, mult in UNIT_MULT:
        if re.search(r"(in|amounts? in)\s+(rs\.?|inr)?\s*" + word, header):
            return mult
    return 1


def _amount_after(unit, keyword_rx, window=200):
    """First money amount after a keyword; skips share counts and face values."""
    text = _flat(unit["text"])
    for km in re.finditer(keyword_rx, text, re.IGNORECASE):
        tail = text[km.end():km.end() + window]
        for m in NUMBER_RX.finditer(tail):
            after = tail[m.end():m.end() + 20].lower()
            if re.match(r"\s*(equity\s+)?shares|\s*each", after) or re.match(r"\s*-\s*\d{2}\b", after):
                continue
            if re.fullmatch(r"(19|20)\d\d", m.group(2)):  # a year
                continue
            value = float(m.group(2).replace(",", ""))
            if m.group(3):
                word = m.group(3).strip().lower()
                value *= next(mult for w, mult in UNIT_MULT if word.startswith(w))
            elif not m.group(1):
                value *= _unit_multiplier(unit["text"])
            return str(int(round(value)))
    return None


def extract(units) -> dict:
    """Return {field_id: {"answer": str|None, "source": {"file", "page"}|None}}."""
    out = {}

    def put(fid, answer, unit=None):
        out[fid] = {"answer": answer, "source": _src(unit) if unit and answer is not None else None}

    def first(fn):
        for u in units:
            val = fn(u)
            if val is not None:
                return val, u
        return None, None

    # CIN: first CIN that is not introduced as another company's (holding / subsidiary / associate)
    def cin(u):
        text = _flat(u["text"])
        for m in CIN_RX.finditer(text):
            if not re.search(r"holding|subsidiary|parent|associate", text[max(0, m.start() - 80):m.start()], re.I):
                return m.group(0)
        return None
    put("cin", *first(cin))

    put("email", *first(lambda u: (EMAIL_RX.search(u["text"]) or [None])[0]))

    def name(u):
        if u["page"] != 1:
            return None
        m = re.search(r"^\s*(.*\b(?:Limited|Ltd\.?))\s*$", u["text"], re.M)
        return m.group(1).strip() if m else None
    put("company_name", *first(name))

    put("registered_office", *first(lambda u: (lambda m: m.group(1).strip() if m else None)(
        re.search(r"(?:Registered|Regd\.?)\s+office(?:\s+address)?\s*:\s*(.+)", u["text"], re.I))))

    put("industry", *first(lambda u: (lambda m: m.group(1).strip() if m else None)(
        re.search(r"engaged in (?:the business of )?([^.]+)\.", _flat(u["text"]), re.I))))

    # financial year
    def fy_range(u):
        text = _flat(u["text"])
        m = re.search(r"(" + DATE_RX.pattern + r")\s+to\s+(" + DATE_RX.pattern + r")", text, re.I)
        return (m.group(1), m.group(2)) if m else None
    rng, rng_unit = first(fy_range)

    def year_ended(u):
        m = re.search(r"year ended\s+(" + DATE_RX.pattern + r")", _flat(u["text"]), re.I)
        return m.group(1) if m else None
    ended, ended_unit = first(year_ended)
    if ended:
        put("fy_to", ended, ended_unit)
    elif rng:
        put("fy_to", rng[1], rng_unit)
    else:
        put("fy_to", None)
    if rng:
        put("fy_from", rng[0], rng_unit)
    elif ended and parse_date(ended):
        start = parse_date(ended).replace(year=parse_date(ended).year - 1) + timedelta(days=1)
        put("fy_from", start.strftime("%d/%m/%Y"), ended_unit)
    else:
        put("fy_from", None)

    def sentence_date(u, must):
        for s in _sentences(u["text"]):
            if all(re.search(p, s, re.I) for p in must):
                d = _dates(s)
                if d:
                    return d[0]
        return None
    put("board_meeting_date", *first(lambda u: sentence_date(u, [r"\bboard\b", r"approv"])))
    put("agm_date", *first(lambda u: sentence_date(u, [r"annual general meeting|\bAGM\b"])))

    put("authorised_capital", *first(lambda u: _amount_after(u, r"authori[sz]ed")))
    put("revenue", *first(lambda u: _amount_after(u, r"revenue from operations", window=60)))
    put("profit_after_tax", *first(lambda u: _amount_after(u, r"profit after tax|profit for the year", window=60)))

    # yes / no questions: whole-text keyword searches, first unit with a decisive match
    def yes_no(neg_rx, pos_rx):
        for u in units:
            t = _flat(u["text"])
            if re.search(neg_rx, t, re.I):
                return "No", u
        for u in units:
            if re.search(pos_rx, _flat(u["text"]), re.I):
                return "Yes", u
        return "No", None

    put("consolidated_fs", *yes_no(
        r"consolidat[^.]{0,80}(not required|not applicable|does not arise)|(not required|no subsidiar)[^.]{0,80}consolidat",
        r"consolidated financial statements"))
    put("secretarial_audit", *yes_no(
        r"secretarial audit[^.]{0,60}not applicable|not applicable[^.]{0,60}secretarial audit",
        r"secretarial audit"))
    put("cag_comments", *yes_no(r"(?!x)x", r"(issued|has made|with) comments|comments under section 143\(6\)\(b\)"))
    put("cag_supplementary_audit", *yes_no(r"(?!x)x", r"(has |have )?conducted a supplementary audit"))
    return out
