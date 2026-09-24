#!/usr/bin/env python3
"""
Builds the synthetic form-filling benchmark: six fictional one-page forms with different layouts.

    python benchmarks/generate_form_benchmark.py [--out benchmarks/data/forms]

For each form <id>/ holds
    empty.pdf        the blank form
    golden.pdf       the same form filled in with sample values (the "golden sample")
    new_data.json    the values to place on the blank form, keyed by a short field name

and ground_truth.json lists, for every form and key, where the new value belongs: the input box
(PDF points, top-left origin, as PyMuPDF and phase1/phase1.py report them) or, for a checkbox
group, the box of every option and which option is chosen.

Layouts
    f1_labels_left      label left of each box, option checkboxes before their text
    f2_labels_above     label above each box
    f3_two_column       two columns of label/box pairs
    f4_yes_no_rows      questionnaire rows with "Yes [ ] No [ ]" (option text before the box)
    f5_ruled_grid       a ruled table: label cells and value cells share grid lines
    f6_comb_kyc         labels above, character-cell ("comb") fields and larger 9 pt checkboxes

All names, numbers and organisations are fictional.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE_W, PAGE_H = 595.0, 842.0   # A4 in points
CB = 5.0                         # default checkbox side (points)
BOX_H = 20.0
NOTE = "Fictional sample form for benchmarking. The organisation and all values are invented."


def tl(x, y, w, h):
    """reportlab rectangle (bottom-left origin) -> [x0, y0, x1, y1] with a top-left origin."""
    return [round(x, 2), round(PAGE_H - y - h, 2), round(x + w, 2), round(PAGE_H - y, 2)]


# --------------------------------------------------------------------------- layout builders
# Each field: {"key", "kind": text|checkbox|comb, "label": (text, x, y), "golden", "new", ...}
#   text:     "box": (x, y, w, h)
#   comb:     "cells": [(x, y, w, h), ...]
#   checkbox: "options": [{"text", "box": (x, y, s), "text_at": (x, y)}]


def text_field(key, label, lx, ly, box, golden, new):
    return {"key": key, "kind": "text", "label": (label, lx, ly), "box": box, "golden": golden, "new": new}


def check_group(key, label, lx, ly, options, golden, new, side=CB, text_first=False, gap=4.0, font=10):
    """options: [(text, x)] at baseline ly; box before the text unless text_first."""
    opts = []
    for text, x in options:
        if text_first:
            tw = _text_width(text, font)
            opts.append({"text": text, "text_at": (x, ly), "box": (x + tw + gap, ly, side)})
        else:
            opts.append({"text": text, "box": (x, ly, side), "text_at": (x + side + gap, ly)})
    return {"key": key, "kind": "checkbox", "label": (label, lx, ly), "options": opts,
            "golden": golden, "new": new}


def comb_field(key, label, lx, ly, x, y, n, cell_w, cell_h, golden, new, groups=None):
    cells, cx = [], x
    for i in range(n):
        cells.append((cx, y, cell_w, cell_h))
        cx += cell_w
        if groups and (i + 1) in groups:
            cx += 6
    return {"key": key, "kind": "comb", "label": (label, lx, ly), "cells": cells, "golden": golden, "new": new}


def _text_width(text, size=10, font="Helvetica"):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    return stringWidth(text, font, size)


def f1_labels_left():
    rows = [("Applicant name", "1. Applicant name:", "Meera Example", "Jordan Sample", 300),
            ("Date of birth", "2. Date of birth (DD/MM/YYYY):", "03/02/1991", "17/10/1986", 120),
            ("Address", "3. Address:", "12 Sample Street, Examplepur", "4 Demo Lane, Sampleton", 330),
            ("Phone number", "4. Phone number:", "+91 90000 11111", "+91 90000 22222", 160),
            ("E-mail", "5. E-mail:", "meera@example.invalid", "jordan@example.invalid", 250)]
    fields, y = [], 730
    for key, label, g, n, w in rows:
        fields.append(text_field(key, label, 50, y + 6, (215, y, w, BOX_H), g, n))
        y -= 38
    fields.append(check_group("Membership type", "6. Membership type:", 50, y + 6,
                              [("Annual", 215), ("Life", 290), ("Student", 350)], "Life", "Student"))
    y -= 32
    fields.append(check_group("Preferred language", "7. Preferred language:", 50, y + 6,
                              [("English", 215), ("Hindi", 290), ("Marathi", 350)], "English", "Hindi"))
    return {"id": "f1_labels_left", "title": "LIBRARY MEMBERSHIP APPLICATION",
            "description": "label left of each box; checkbox before its option text", "fields": fields}


def f2_labels_above():
    rows = [("Full name", "Full name", "Ravi Placeholder", "Sam Specimen"),
            ("Organisation", "Organisation", "Example Instruments Co.", "Sample Textiles Co."),
            ("Purpose of visit", "Purpose of visit", "Equipment audit", "Vendor meeting"),
            ("Host employee", "Host employee", "N. Demo", "K. Testcase"),
            ("Vehicle number", "Vehicle number", "ZZ 01 AB 1234", "ZZ 09 CD 5678")]
    fields, y = [], 700
    for key, label, g, n in rows:
        fields.append(text_field(key, label, 50, y + BOX_H + 6, (50, y, 400, BOX_H), g, n))
        y -= 52
    grp = check_group("ID proof shown", "ID proof shown", 50, y + 4,
                      [("Passport", 50), ("Driving licence", 140), ("Voter ID", 260)], "Passport", "Voter ID")
    grp["label"] = ("ID proof shown", 50, y + BOX_H + 6)   # label above, options on the line below it
    fields.append(grp)
    return {"id": "f2_labels_above", "title": "VISITOR REGISTRATION FORM",
            "description": "label above each box; checkbox options on the line below the label", "fields": fields}


def f3_two_column():
    left = [("First name", "First name:", "Asha", "Leo"), ("Employee ID", "Employee ID:", "EX-1042", "EX-2210"),
            ("Joining date", "Joining date:", "01/07/2024", "15/01/2025"), ("City", "City:", "Examplepur", "Sampleton")]
    right = [("Last name", "Last name:", "Example", "Specimen"), ("Department", "Department:", "Finance", "Logistics"),
             ("Manager", "Manager:", "P. Demo", "R. Mockley"), ("PIN code", "PIN code:", "999101", "999202")]
    fields, y = [], 720
    for (lk, ll, lg, ln), (rk, rl, rg, rn) in zip(left, right):
        fields.append(text_field(lk, ll, 40, y + 6, (125, y, 140, BOX_H), lg, ln))
        fields.append(text_field(rk, rl, 300, y + 6, (385, y, 160, BOX_H), rg, rn))
        y -= 36
    fields.append(check_group("Shift", "Shift:", 40, y + 6, [("Day", 125), ("Night", 185)], "Day", "Night"))
    fields.append(check_group("Laptop required", "Laptop required:", 300, y + 6, [("Yes", 385), ("No", 440)],
                              "Yes", "No"))
    return {"id": "f3_two_column", "title": "EMPLOYEE ONBOARDING DETAILS",
            "description": "two columns of label/box pairs, checkbox groups in both columns", "fields": fields}


def f4_yes_no_rows():
    fields, y = [], 730
    for key, label, g, n, w in [("Name", "Name:", "Tara Example", "Omar Sample", 250),
                                ("Policy number", "Policy number:", "POL-000123", "POL-000987", 150),
                                ("Height (cm)", "Height (cm):", "168", "181", 60),
                                ("Weight (kg)", "Weight (kg):", "61", "77", 60)]:
        fields.append(text_field(key, label, 50, y + 6, (160, y, w, BOX_H), g, n))
        y -= 34
    y -= 10
    questions = [("Do you smoke", "a) Do you smoke?", "No", "Yes"),
                 ("Do you have diabetes", "b) Do you have diabetes?", "No", "No"),
                 ("Hospitalised in the last 5 years", "c) Have you been hospitalised in the last 5 years?", "Yes", "No"),
                 ("Do you consume alcohol", "d) Do you consume alcohol?", "Yes", "Yes")]
    for key, label, g, n in questions:
        fields.append(check_group(key, label, 50, y, [("Yes", 400), ("No", 460)], g, n, text_first=True))
        y -= 26
    return {"id": "f4_yes_no_rows", "title": "HEALTH DECLARATION",
            "description": "text fields plus yes/no questionnaire rows with the option text before its box",
            "fields": fields}


def f5_ruled_grid():
    pairs = [("Vendor name", "Example Fasteners Co.", "Sample Paints Co."), ("Tax registration no.", "ZZ-TAX-1111", "ZZ-TAX-2222"),
             ("Contact person", "V. Example", "M. Sample"), ("Phone", "+91 90000 33333", "+91 90000 44444"),
             ("Bank name", "Example Co-op Bank", "Sample Rural Bank"), ("Account number", "000111222333", "000444555666"),
             ("Branch code", "EXMP0001234", "SMPL0005678"), ("City", "Demoabad", "Mockpur")]
    x0, top, cw, ch = 40, 720, 128.75, 26
    fields = []
    for i, (label, g, n) in enumerate(pairs):
        row, col = divmod(i, 2)
        y = top - (row + 1) * ch
        lx = x0 + col * 2 * cw
        fields.append(text_field(label, label, lx + 4, y + 9, (lx + cw, y, cw, ch), g, n))
    y = top - 5 * ch
    grp = check_group("Vendor type", "Vendor type", x0 + 4, y + 9,
                      [("Manufacturer", x0 + cw + 8), ("Trader", x0 + cw + 110), ("Service provider", x0 + cw + 190)],
                      "Trader", "Service provider")
    fields.append(grp)
    return {"id": "f5_ruled_grid", "title": "VENDOR REGISTRATION",
            "description": "ruled table: label cells and value cells share grid lines",
            "grid": {"x0": x0, "top": top, "cw": cw, "ch": ch, "rows": 5, "cols": 4},
            "fields": fields}


def f6_comb_kyc():
    fields = []
    fields.append(text_field("Customer name", "Customer name", 40, 726, (40, 700, 330, BOX_H), "Neha Example", "Arjun Sample"))
    fields.append(comb_field("Date of birth", "Date of birth (DD MM YYYY)", 390, 726, 390, 700, 8, 16, BOX_H,
                             "09/03/1990", "21/11/1985", groups=(2, 4)))
    fields.append(text_field("Father's or spouse's name", "Father's or spouse's name", 40, 676, (40, 650, 330, BOX_H),
                             "R. Example", "S. Sample"))
    fields.append(text_field("Mobile", "Mobile", 390, 676, (390, 650, 165, BOX_H), "9000055555", "9000066666"))
    fields.append(text_field("E-mail", "E-mail", 40, 626, (40, 600, 250, BOX_H), "neha@example.invalid", "arjun@example.invalid"))
    fields.append(text_field("Occupation", "Occupation", 310, 626, (310, 600, 120, BOX_H), "Engineer", "Teacher"))
    fields.append(text_field("Annual income", "Annual income (Rs.)", 445, 626, (445, 600, 110, BOX_H), "900000", "650000"))
    fields.append(text_field("Address", "Address", 40, 576, (40, 550, 390, BOX_H), "8 Example Road, Testnagar", "3 Sample Nagar, Mockpur"))
    fields.append(comb_field("PIN code", "PIN code", 445, 576, 445, 550, 6, 18, BOX_H, "999303", "999404"))
    fields.append(comb_field("Tax ID", "Tax ID (10 characters)", 40, 526, 40, 500, 10, 18, BOX_H, "ABCDE1234F", "PQRSX6789K"))
    fields.append(check_group("Account type", "Account type", 280, 516, [("Savings", 360), ("Current", 440)],
                              "Savings", "Current", side=9.0))
    fields.append(check_group("Nationality", "Nationality", 40, 466, [("Indian", 120), ("Other", 200)],
                              "Indian", "Other", side=9.0))
    return {"id": "f6_comb_kyc", "title": "SAVINGS ACCOUNT OPENING - KYC DETAILS",
            "description": "labels above boxes, character-cell (comb) fields, 9 pt checkboxes", "fields": fields}


LAYOUTS = [f1_labels_left, f2_labels_above, f3_two_column, f4_yes_no_rows, f5_ruled_grid, f6_comb_kyc]


# --------------------------------------------------------------------------- rendering


def draw(form: dict, path: Path, filled: bool) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1)
    c.setTitle(f"{form['title'].title()} (fictional)")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 780, form["title"])
    c.setFont("Helvetica", 8)
    c.drawString(40, 766, NOTE)

    if "grid" in form:
        g = form["grid"]
        c.setLineWidth(0.75)
        width, height = g["cols"] * g["cw"], g["rows"] * g["ch"]
        for r in range(g["rows"] + 1):
            c.line(g["x0"], g["top"] - r * g["ch"], g["x0"] + width, g["top"] - r * g["ch"])
        for col in range(g["cols"] + 1):
            # the last row has one label cell and one value cell spanning three columns
            y_bottom = g["top"] - height if col in (0, 1, g["cols"]) else g["top"] - (g["rows"] - 1) * g["ch"]
            c.line(g["x0"] + col * g["cw"], g["top"], g["x0"] + col * g["cw"], y_bottom)

    for f in form["fields"]:
        label, lx, ly = f["label"]
        c.setFont("Helvetica", 10)
        c.drawString(lx, ly, label)
        c.setLineWidth(0.75)
        if f["kind"] == "text":
            x, y, w, h = f["box"]
            if "grid" not in form:
                c.rect(x, y, w, h, stroke=1, fill=0)
            if filled:
                c.drawString(x + 4, y + (h - 7) / 2, f["golden"])
        elif f["kind"] == "comb":
            for i, (x, y, w, h) in enumerate(f["cells"]):
                c.rect(x, y, w, h, stroke=1, fill=0)
                if filled:
                    ch = [k for k in f["golden"] if k.isalnum()][i]
                    c.drawString(x + (w - _text_width(ch)) / 2, y + (h - 7) / 2, ch)
        else:
            c.setLineWidth(0.5)
            for o in f["options"]:
                x, y, s = o["box"]
                c.rect(x, y, s, s, stroke=1, fill=0)
                c.setFont("Helvetica", 10)
                c.drawString(o["text_at"][0], o["text_at"][1], o["text"])
                if filled and o["text"] == f["golden"]:
                    size = s * 1.3
                    c.setFont("Helvetica-Bold", size)
                    c.drawString(x + (s - _text_width("X", size, "Helvetica-Bold")) / 2, y + s * 0.1, "X")
    c.setFont("Helvetica", 10)
    c.drawString(40, 120, "Signature: ______________________        Date: ______________")
    c.save()


def gold_entry(f: dict) -> dict:
    if f["kind"] == "text":
        return {"key": f["key"], "kind": "text", "value": f["new"], "bbox": tl(*f["box"])}
    if f["kind"] == "comb":
        xs = [cell[0] for cell in f["cells"]]
        x0, y, _, h = f["cells"][0]
        x1 = f["cells"][-1][0] + f["cells"][-1][2]
        return {"key": f["key"], "kind": "comb", "value": f["new"], "bbox": tl(x0, y, x1 - x0, h),
                "cells": [tl(*cell) for cell in f["cells"]], "n_cells": len(xs)}
    return {"key": f["key"], "kind": "checkbox", "value": f["new"],
            "options": [{"value": o["text"], "bbox": tl(o["box"][0], o["box"][1], o["box"][2], o["box"][2])}
                        for o in f["options"]]}


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {"description": "Synthetic form-filling benchmark. Boxes are PDF points with a top-left origin. "
                            "All forms and values are fictional.",
             "page_size": [PAGE_W, PAGE_H], "forms": []}
    for make in LAYOUTS:
        form = make()
        d = out_dir / form["id"]
        d.mkdir(exist_ok=True)
        draw(form, d / "empty.pdf", filled=False)
        draw(form, d / "golden.pdf", filled=True)
        new_data = {f["key"]: f["new"] for f in form["fields"]}
        (d / "new_data.json").write_text(json.dumps(new_data, indent=2, ensure_ascii=False) + "\n")
        truth["forms"].append({"id": form["id"], "title": form["title"], "description": form["description"],
                               "fields": [gold_entry(f) for f in form["fields"]]})
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "data" / "forms"), help="output folder (default: benchmarks/data/forms)")
    args = ap.parse_args()
    truth = build(Path(args.out))
    n = sum(len(f["fields"]) for f in truth["forms"])
    print(f"{len(truth['forms'])} forms, {n} fields -> {args.out}")


if __name__ == "__main__":
    main()
