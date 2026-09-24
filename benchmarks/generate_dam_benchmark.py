#!/usr/bin/env python3
"""
Builds the synthetic DAM field-extraction benchmark: six fictional listed companies, each with a
folder of three documents written from a ground-truth record.

    python benchmarks/generate_dam_benchmark.py [--out benchmarks/data/dam]

Per company:
    financial_statements.pdf   6 pages: cover, balance sheet, profit and loss, notes, accounting policies,
                               related parties and other notes
    directors_report.docx      board meetings, subsidiaries, audits, AGM, contact details and the
                               usual boilerplate sections (MD&A, committees, auditors, deposits, ...)
    trial_balance.xlsx         company master sheet and current / prior-year trial balances

Facts are spread across the files (the AGM date is only in the directors' report, the authorised
capital only in the notes, one e-mail only in the workbook) and the documents carry distractors:
prior-year figures and authorised capital, earlier board meeting and AGM dates, a holding
company's CIN, a share registrar's e-mail and a subsidiary for which secretarial audit does not
apply. Amounts are shown in rupees, lakhs, crores or millions depending on the company, and date
formats differ between companies. Two fields are not stated anywhere (gold value null).

ground_truth.json records, for every company and field, the gold value and every (file, page)
where the value is stated; the script checks each of those locations against the text extracted
from the generated files before writing it. Company names, identifiers, addresses, people and
figures are all invented. CINs use the non-existent state code ZZ.
"""

import argparse
import json
import re
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dam_fields import FIELDS  # noqa: E402
from plain_text import docx_text  # noqa: E402

FOOTER = "Synthetic benchmark document. The company, its people and all figures are fictional."
FIXED_TIME = datetime(2026, 1, 1, 0, 0, 0)

PDF_NAME, DOCX_NAME, XLSX_NAME = "financial_statements.pdf", "directors_report.docx", "trial_balance.xlsx"

# --------------------------------------------------------------------------- formatting helpers


def inr(n: int) -> str:
    """Indian digit grouping: 12345678 -> 1,23,45,678."""
    s = str(abs(int(n)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + s


UNITS = {
    "rupees": (1, "Amount in Rs."),
    "lakhs": (100_000, "All amounts in Rs. lakhs unless otherwise stated"),
    "crore": (10_000_000, "(Rs. in crore)"),
    "millions": (1_000_000, "All amounts in INR millions"),
}


def in_unit(rupees: int, unit: str) -> str:
    div = UNITS[unit][0]
    if div == 1:
        return inr(rupees)
    if (rupees * 100) % div:
        raise ValueError(f"{rupees} is not exact to two decimals in {unit}")
    hundredths = rupees * 100 // div
    whole, frac = divmod(hundredths, 100)
    grouped = f"{whole:,}" if unit == "millions" else inr(whole)
    return f"{grouped}.{frac:02d}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt_date(d: date, style: str) -> str:
    if style == "dmy_slash":
        return d.strftime("%d/%m/%Y")
    if style == "dmy_dot":
        return d.strftime("%d.%m.%Y")
    if style == "dmy_dash":
        return d.strftime("%d-%m-%Y")
    if style == "long":                      # 14 May 2024
        return f"{d.day} {d.strftime('%B %Y')}"
    if style == "ordinal":                   # 22nd May, 2024
        return f"{ordinal(d.day)} {d.strftime('%B')}, {d.year}"
    if style == "us":                        # July 29, 2024
        return f"{d.strftime('%B')} {d.day}, {d.year}"
    raise ValueError(style)


def ddmmyyyy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def D(s: str) -> date:
    return datetime.strptime(s, "%d/%m/%Y").date()


# --------------------------------------------------------------------------- the companies

COMPANIES = [
    {
        "id": "c1_velmora", "name": "Velmora Pumps and Valves Limited", "cin": "L29120ZZ1994PLC011872",
        "address": "Plot 14, Sector 9, Examplepur Industrial Area, Examplepur, Zedland 999014",
        "email": "secretarial@velmora-pumps.example", "email_in": ["pdf_cover", "docx"],
        "rta_email": "investors@sampledesk-registrars.example",
        "fy_from": "01/04/2023", "fy_to": "31/03/2024", "fy_from_in": "pdf_cover",
        "board": "14/05/2024", "board_list": ["12/05/2023", "08/08/2023", "07/11/2023", "06/02/2024"],
        "prev_board": "12/05/2023",
        "agm": "26/07/2024", "agm_no": 30, "prev_agm": "28/07/2023",
        "auth": 250_000_000, "auth_prev": 250_000_000, "issued": 184_000_000,
        "industry": "Manufacture of industrial pumps and valves",
        "unit": "rupees", "date_style": "long", "docx_date_style": "dmy_slash",
        "revenue": 1_846_235_000, "revenue_prev": 1_612_048_000,
        "pat": 148_216_000, "pat_prev": 110_574_000,
        "consolidated": False, "cag": None, "subsidiaries": [], "holding": None,
        "consol_text": "The Company does not have any subsidiary, joint venture or associate company. Hence, "
                       "consolidated financial statements are not required to be prepared.",
        "cin_label": "Corporate Identity Number", "office_label": "Registered Office",
    },
    {
        "id": "c2_tarnwick", "name": "Tarnwick Logistics Limited", "cin": "L63090ZZ2006PLC048213",
        "address": "7th Floor, Harbour View Towers, Dock Road, Sampleton, Zedland 999221",
        "email": "cs@tarnwick-logistics.example", "email_in": ["xlsx"],
        "rta_email": "helpdesk@ledgerline-rta.example",
        "fy_from": "01/04/2023", "fy_to": "31/03/2024", "fy_from_in": "xlsx",
        "board": "22/05/2024", "board_list": ["19/05/2023", "10/08/2023", "09/11/2023", "12/02/2024"],
        "prev_board": "19/05/2023",
        "agm": "20/08/2024", "agm_no": 18, "prev_agm": "23/08/2023",
        "auth": 500_000_000, "auth_prev": 300_000_000, "issued": 262_500_000,
        "industry": "Freight forwarding and warehousing services",
        "unit": "lakhs", "date_style": "ordinal", "docx_date_style": "ordinal",
        "revenue": 4_231_864_000, "revenue_prev": 3_790_215_000,
        "pat": 210_637_000, "pat_prev": 174_490_000,
        "consolidated": True, "cag": None,
        "subsidiaries": ["Tarnwick Cold Chain Limited", "Tarnwick Air Cargo Limited"], "holding": None,
        "consol_text": "The Company has two wholly owned subsidiaries, Tarnwick Cold Chain Limited and Tarnwick "
                       "Air Cargo Limited. The consolidated financial statements of the Company and its "
                       "subsidiaries form part of this Annual Report and are being filed along with the "
                       "standalone financial statements.",
        "secretarial_distractor": "Secretarial audit is not applicable to Tarnwick Air Cargo Limited, a wholly "
                                  "owned subsidiary, as it does not meet the thresholds under section 204.",
        "cin_label": "CIN", "office_label": "Regd. Office",
    },
    {
        "id": "c3_zedpower", "name": "Zedland State Power Generation Corporation Limited",
        "cin": "L40101ZZ1987SGC004512",
        "address": "Urja Bhavan, 3 Grid Lane, Testnagar, Zedland 999402",
        "email": "cosec@zspgcl.example", "email_in": ["pdf_cover"],
        "rta_email": "rta@sampledesk-registrars.example",
        "fy_from": "01/04/2023", "fy_to": "31/03/2024", "fy_from_in": "docx",
        "board": "29/07/2024", "board_list": ["28/07/2023", "13/11/2023", "14/02/2024", "27/03/2024"],
        "prev_board": "28/07/2023",
        "agm": "27/09/2024", "agm_no": 37, "prev_agm": "28/09/2023",
        "auth": 50_000_000_000, "auth_prev": 50_000_000_000, "issued": 18_512_000_000,
        "industry": "Generation of electricity from thermal and hydro power plants",
        "unit": "crore", "date_style": "us", "docx_date_style": "us",
        "revenue": 84_123_500_000, "revenue_prev": 79_551_000_000,
        "pat": 3_124_800_000, "pat_prev": 2_800_200_000,
        "consolidated": False, "subsidiaries": [], "holding": None,
        "cag": {"comments": True, "supplementary": True,
                "text": "The Comptroller and Auditor General of India (C&AG) conducted a supplementary audit of "
                        "the financial statements under section 143(6)(a) of the Act and issued comments under "
                        "section 143(6)(b), which are annexed to this report together with the replies of the "
                        "management."},
        "consol_text": "The Corporation has no subsidiaries; consolidated financial statements are therefore not "
                       "applicable.",
        "cin_label": "CIN", "office_label": "Registered office address",
    },
    {
        "id": "c4_quorrin", "name": "Quorrin Textiles Limited", "cin": "L17110ZZ2011PLC063347",
        "address": "Survey No. 41, Loom Street, Mockpur, Zedland 999318",
        "email": None, "email_in": [],
        "rta_email": "grievances@ledgerline-rta.example",
        "fy_from": "01/04/2024", "fy_to": "31/03/2025", "fy_from_in": "pdf_cover",
        "board": "12/05/2025", "board_list": ["15/05/2024", "09/08/2024", "12/11/2024", "11/02/2025"],
        "prev_board": "15/05/2024",
        "agm": "17/09/2025", "agm_no": 14, "prev_agm": "20/09/2024",
        "auth": 400_000_000, "auth_prev": 200_000_000, "issued": 316_000_000,
        "industry": "Spinning and weaving of cotton yarn and fabrics",
        "unit": "rupees", "date_style": "dmy_dot", "docx_date_style": "dmy_dot",
        "revenue": 2_964_180_000, "revenue_prev": 2_739_905_000,
        "pat": 98_640_000, "pat_prev": 123_110_000,
        "consolidated": False, "subsidiaries": [], "holding": None,
        "cag": {"comments": False, "supplementary": False,
                "text": "The Company is not a Government company; the provisions of section 143(5) and 143(6) "
                        "of the Act relating to the Comptroller and Auditor General of India do not apply."},
        "consol_text": "As the Company has no subsidiary or associate company, the requirement of consolidation "
                       "does not arise.",
        "cin_label": "Corporate Identity Number (CIN)", "office_label": "Registered Office",
    },
    {
        "id": "c5_ambrel", "name": "Ambrel Agro Foods Limited", "cin": "L15490ZZ1999GOI021905",
        "address": "Krishi Complex, 22 Mandi Road, Demoabad, Zedland 999507",
        "email": "company.secretary@ambrel-agro.example", "email_in": ["docx"],
        "rta_email": "support@sampledesk-registrars.example",
        "fy_from": "01/04/2023", "fy_to": "31/03/2024", "fy_from_in": "xlsx",
        "board": "06/08/2024", "board_list": ["03/08/2023", "09/11/2023", "08/02/2024"],
        "prev_board": "03/08/2023",
        "agm": "25/09/2024", "agm_no": 25, "prev_agm": "27/09/2023",
        "auth": 150_000_000, "auth_prev": 150_000_000, "issued": 112_450_000,
        "industry": "Processing and packaging of edible oils and pulses",
        "unit": "lakhs", "date_style": "dmy_dash", "docx_date_style": "dmy_dash",
        "revenue": 6_124_830_000, "revenue_prev": 5_900_211_000,
        "pat": 184_070_000, "pat_prev": 160_255_000,
        "consolidated": True, "subsidiaries": ["Ambrel Cold Storage Limited"], "holding": None,
        "cag": {"comments": False, "supplementary": False,
                "text": "The Comptroller and Auditor General of India, vide letter dated 02-09-2024, has decided "
                        "not to conduct a supplementary audit of the financial statements under section 143(6)(a) "
                        "of the Act and has no comments to offer on the auditors' report."},
        "consol_text": "In accordance with section 129(3) of the Act, the Company has prepared consolidated "
                       "financial statements including its subsidiary, Ambrel Cold Storage Limited, which are "
                       "filed together with these standalone financial statements.",
        "cin_label": "CIN", "office_label": "Registered Office",
    },
    {
        "id": "c6_lumeqa", "name": "Lumeqa Software Limited", "cin": "L72200ZZ2015PLC079431",
        "address": "Unit 502, Fifth Floor, Techpark Phase II, Fictionganj, Zedland 999633",
        "email": "investor.relations@lumeqa.example", "email_in": ["pdf_cover", "docx"],
        "rta_email": "desk@ledgerline-rta.example",
        "fy_from": "01/04/2023", "fy_to": "31/03/2024", "fy_from_in": "pdf_cover",
        "board": "25/04/2024", "board_list": ["27/04/2023", "21/07/2023", "20/10/2023", "19/01/2024"],
        "prev_board": "27/04/2023",
        "agm": None, "agm_no": 9, "prev_agm": "29/06/2023",
        "auth": 120_000_000, "auth_prev": 120_000_000, "issued": 97_300_000,
        "industry": "Development of software products for logistics companies",
        "unit": "millions", "date_style": "dmy_slash", "docx_date_style": "dmy_slash",
        "revenue": 973_160_000, "revenue_prev": 840_930_000,
        "pat": 216_410_000, "pat_prev": 179_830_000,
        "consolidated": False, "subsidiaries": [], "cag": None,
        "holding": {"name": "Kestrow Holdings Limited", "cin": "L65990ZZ1995PLC012345"},
        "consol_text": "The Company is a subsidiary of Kestrow Holdings Limited and has no subsidiary of its own; "
                       "accordingly, it is not required to prepare consolidated financial statements. The "
                       "consolidated financial statements of the group are prepared by Kestrow Holdings Limited.",
        "cin_label": "CIN", "office_label": "Registered Office",
    },
]

# --------------------------------------------------------------------------- derived figures


def figures(c: dict, prior: bool) -> dict:
    """A consistent set of statement lines (in rupees) for the current or prior year."""
    rev = c["revenue_prev"] if prior else c["revenue"]
    pat = c["pat_prev"] if prior else c["pat"]
    div = UNITS[c["unit"]][0]
    step = max(div // 100, 1)

    def r(x):  # round to something representable to two decimals in the display unit
        return int(round(x / step)) * step

    other_income = r(rev * 0.012)
    tax = r(pat / 3)
    pbt = pat + tax
    expenses = rev + other_income - pbt
    capital = c["issued"]
    borrowings = r(rev * 0.18)
    payables = r(rev * 0.11)
    ppe = r(rev * 0.42)
    inventories = r(rev * 0.14)
    receivables = r(rev * 0.16)
    cash = r(rev * 0.05)
    total_assets = ppe + inventories + receivables + cash
    other_equity = total_assets - capital - borrowings - payables
    return {"revenue": rev, "other_income": other_income, "total_income": rev + other_income,
            "expenses": expenses, "pbt": pbt, "tax": tax, "pat": pat,
            "capital": capital, "other_equity": other_equity, "borrowings": borrowings, "payables": payables,
            "total_liabilities": capital + other_equity + borrowings + payables,
            "ppe": ppe, "inventories": inventories, "receivables": receivables, "cash": cash,
            "total_assets": total_assets}


def _round_to(x: float, c: dict) -> int:
    """Round a rupee amount so that it is exact to two decimals in the company's display unit."""
    step = max(UNITS[c["unit"]][0] // 100, 1)
    return int(round(x / step)) * step


ACCOUNTING_POLICIES = [
    "Basis of preparation. The financial statements have been prepared in accordance with the Indian Accounting "
    "Standards (Ind AS) notified under section 133 of the Companies Act, 2013, on the historical cost basis except "
    "for certain financial instruments measured at fair value. All assets and liabilities are classified as current "
    "or non-current according to the operating cycle of twelve months.",
    "Revenue recognition. Revenue from contracts with customers is recognised when control of the goods or services "
    "is transferred to the customer at an amount that reflects the consideration to which the Company expects to be "
    "entitled. Revenue is measured net of returns, trade discounts and goods and services tax. Interest income is "
    "recognised using the effective interest method.",
    "Property, plant and equipment. Items of property, plant and equipment are stated at cost less accumulated "
    "depreciation and impairment losses. Depreciation is provided on the straight-line method over the useful lives "
    "prescribed in Schedule II to the Act, except where technical assessment supports a different life. Capital "
    "work-in-progress is stated at cost.",
    "Inventories. Inventories are valued at the lower of cost and net realisable value. Cost is determined on a "
    "weighted average basis and includes all costs of purchase, conversion and other costs incurred in bringing the "
    "inventories to their present location and condition.",
    "Employee benefits. Contributions to provident fund are charged to the statement of profit and loss when due. "
    "The gratuity liability is determined by an independent actuary at each balance sheet date using the projected "
    "unit credit method; re-measurement gains and losses are recognised in other comprehensive income.",
    "Income taxes. Current tax is measured at the amount expected to be paid under the Income-tax Act, 1961. "
    "Deferred tax is recognised on temporary differences between the carrying amounts of assets and liabilities and "
    "their tax bases, to the extent that it is probable that taxable profit will be available.",
    "Provisions and contingent liabilities. A provision is recognised when the Company has a present obligation as a "
    "result of a past event and a reliable estimate can be made of the outflow. Contingent liabilities are disclosed "
    "unless the possibility of an outflow is remote.",
    "Financial instruments. Financial assets are classified at initial recognition as measured at amortised cost, at "
    "fair value through other comprehensive income or at fair value through profit or loss. Trade receivables are "
    "assessed for impairment using the simplified expected credit loss approach.",
]

DOCX_BOILERPLATE = [
    ("Management discussion and analysis",
     "The sector in which the Company operates grew steadily during the year, supported by stable input costs and "
     "improving demand from domestic customers. The Company continued to invest in capacity, automation and the "
     "training of its people. Key risks include competition, movements in commodity prices and interest rates, and "
     "delays in collections from customers; the Board reviews these risks every quarter."),
    ("Deposits", "The Company has not accepted any deposits from the public within the meaning of Chapter V of the "
                 "Act during the year, and no amount of principal or interest was outstanding at the end of the year."),
    ("Internal financial controls",
     "The Company has adequate internal financial controls with reference to the financial statements. During the "
     "year such controls were tested and no reportable material weakness in their design or operation was observed."),
    ("Cost records", "The maintenance of cost records under section 148(1) of the Act is not applicable to the "
                     "Company's products for the year under review."),
    ("Vigil mechanism", "The Company has a whistle blower policy under which directors and employees may report "
                        "genuine concerns directly to the chairperson of the Audit Committee."),
    ("Particulars of employees", "The information required under section 197(12) of the Act read with the relevant "
                                 "rules is available for inspection by the members at the registered office."),
]


def fy_label(c: dict, prior: bool = False) -> str:
    start = D(c["fy_from"]).year - (1 if prior else 0)
    return f"FY {start}-{str(start + 1)[-2:]}"


# --------------------------------------------------------------------------- writers


class Placements:
    """Collects where each gold value is stated: {field: [(file, page, surface_text)]}."""

    def __init__(self):
        self.by_field = {}

    def add(self, field, file, page, surface):
        self.by_field.setdefault(field, []).append((file, page, surface))


def write_pdf(c: dict, path: Path, P: Placements) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    s = getSampleStyleSheet()
    ds = c["date_style"]
    fy_to_long = fmt_date(D(c["fy_to"]), "long")
    unit_note = UNITS[c["unit"]][1]
    cur, prev = figures(c, False), figures(c, True)
    ylab, plab = fy_label(c), fy_label(c, prior=True)
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, "#777777"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                       ("ALIGN", (1, 0), (-1, -1), "RIGHT")])
    story = []

    # page 1: cover
    story += [Paragraph(c["name"], s["Title"]), Spacer(1, 6),
              Paragraph(f"{c['cin_label']}: {c['cin']}", s["Normal"]),
              Paragraph(f"{c['office_label']}: {c['address']}", s["Normal"])]
    P.add("company_name", PDF_NAME, 1, c["name"])
    P.add("cin", PDF_NAME, 1, c["cin"])
    P.add("registered_office", PDF_NAME, 1, c["address"])
    if "pdf_cover" in c["email_in"]:
        story.append(Paragraph(f"E-mail: {c['email']}", s["Normal"]))
        P.add("email", PDF_NAME, 1, c["email"])
    story += [Spacer(1, 18), Paragraph(f"Standalone Financial Statements for the year ended {fy_to_long}", s["Heading2"])]
    P.add("fy_to", PDF_NAME, 1, fy_to_long)
    if c["fy_from_in"] == "pdf_cover":
        f, t = fmt_date(D(c["fy_from"]), ds), fmt_date(D(c["fy_to"]), ds)
        story.append(Paragraph(f"Financial year covered: {f} to {t}", s["Normal"]))
        P.add("fy_from", PDF_NAME, 1, f)
        P.add("fy_to", PDF_NAME, 1, t)
    story.append(PageBreak())

    # page 2: balance sheet
    story += [Paragraph(f"{c['name']} - Balance Sheet as at {fy_to_long}", s["Heading2"]),
              Paragraph(unit_note, s["Italic"]), Spacer(1, 6)]
    u = lambda v: in_unit(v, c["unit"])  # noqa: E731
    rows = [["Particulars", f"As at end of {ylab}", f"As at end of {plab}"],
            ["Equity share capital", u(cur["capital"]), u(prev["capital"])],
            ["Other equity", u(cur["other_equity"]), u(prev["other_equity"])],
            ["Borrowings", u(cur["borrowings"]), u(prev["borrowings"])],
            ["Trade payables", u(cur["payables"]), u(prev["payables"])],
            ["Total equity and liabilities", u(cur["total_liabilities"]), u(prev["total_liabilities"])],
            ["Property, plant and equipment", u(cur["ppe"]), u(prev["ppe"])],
            ["Inventories", u(cur["inventories"]), u(prev["inventories"])],
            ["Trade receivables", u(cur["receivables"]), u(prev["receivables"])],
            ["Cash and cash equivalents", u(cur["cash"]), u(prev["cash"])],
            ["Total assets", u(cur["total_assets"]), u(prev["total_assets"])]]
    story += [Table(rows, style=grid), PageBreak()]
    P.add("company_name", PDF_NAME, 2, c["name"])

    # page 3: statement of profit and loss
    story += [Paragraph(f"{c['name']} - Statement of Profit and Loss for the year ended {fy_to_long}", s["Heading2"]),
              Paragraph(unit_note, s["Italic"]), Spacer(1, 6)]
    rows = [["Particulars", ylab, plab],
            ["Revenue from operations", u(cur["revenue"]), u(prev["revenue"])],
            ["Other income", u(cur["other_income"]), u(prev["other_income"])],
            ["Total income", u(cur["total_income"]), u(prev["total_income"])],
            ["Total expenses", u(cur["expenses"]), u(prev["expenses"])],
            ["Profit before tax", u(cur["pbt"]), u(prev["pbt"])],
            ["Tax expense", u(cur["tax"]), u(prev["tax"])],
            ["Profit after tax", u(cur["pat"]), u(prev["pat"])]]
    story += [Table(rows, style=grid), PageBreak()]
    P.add("company_name", PDF_NAME, 3, c["name"])
    P.add("revenue", PDF_NAME, 3, u(cur["revenue"]))
    P.add("profit_after_tax", PDF_NAME, 3, u(cur["pat"]))

    # page 4: notes
    story.append(Paragraph("Notes to the financial statements", s["Heading2"]))
    note1 = (f"1. Corporate information. {c['name']} ('the Company') is a public limited company incorporated in "
             f"India and listed on a stock exchange. The Company is engaged in {c['industry'][0].lower()}{c['industry'][1:]}.")
    govt = c["cin"][12:15] in ("GOI", "SGC")
    note1 += (" The Company is a Government company within the meaning of section 2(45) of the Companies Act, 2013."
              if govt else
              " The Company is not a Government company within the meaning of section 2(45) of the Companies Act, 2013.")
    if c["holding"]:
        note1 += (f" The Company is a subsidiary of {c['holding']['name']} "
                  f"(CIN: {c['holding']['cin']}), which holds 64.2% of its equity share capital.")
    story.append(Paragraph(note1, s["Normal"]))
    P.add("industry", PDF_NAME, 4, f"{c['industry'][0].lower()}{c['industry'][1:]}")
    story.append(Spacer(1, 8))

    shares_auth = c["auth"] // 10
    if c["unit"] == "rupees":
        auth_s, auth_prev_s = f"Rs. {inr(c['auth'])}", f"Rs. {inr(c['auth_prev'])}"
    elif c["unit"] == "crore":
        auth_s, auth_prev_s = f"Rs. {inr(c['auth'] // 10_000_000)} crore", f"Rs. {inr(c['auth_prev'] // 10_000_000)} crore"
    elif c["unit"] == "millions":
        auth_s, auth_prev_s = f"INR {c['auth'] // 1_000_000:,} million", f"INR {c['auth_prev'] // 1_000_000:,} million"
    else:  # lakhs: in the table, in the statement unit
        auth_s, auth_prev_s = in_unit(c["auth"], "lakhs"), in_unit(c["auth_prev"], "lakhs")

    if c["unit"] == "lakhs" and c["auth"] != c["auth_prev"]:
        story.append(Paragraph("2. Share capital (Rs. in lakhs)", s["Normal"]))
        story.append(Table([["Particulars", f"As at end of {ylab}", f"As at end of {plab}"],
                            [f"Authorised: {inr(shares_auth)} equity shares of Rs. 10 each", auth_s, auth_prev_s],
                            ["Issued, subscribed and fully paid up", u(cur["capital"]), u(prev["capital"])]],
                           style=grid))
        P.add("authorised_capital", PDF_NAME, 4, auth_s)
    elif c["unit"] == "lakhs":
        story.append(Paragraph(
            f"2. Share capital. The authorised share capital of the Company is Rs. {inr(c['auth'])} divided into "
            f"{inr(shares_auth)} equity shares of Rs. 10 each. The issued, subscribed and paid-up capital is "
            f"Rs. {inr(c['issued'])}.", s["Normal"]))
        P.add("authorised_capital", PDF_NAME, 4, f"Rs. {inr(c['auth'])}")
    elif c["auth"] != c["auth_prev"]:
        story.append(Paragraph(
            f"2. Share capital. During the year the authorised share capital of the Company was increased from "
            f"{auth_prev_s} to {auth_s} by an ordinary resolution of the members. The authorised capital at the "
            f"end of the year is therefore {auth_s}, divided into {inr(shares_auth)} equity shares of Rs. 10 each. "
            f"Issued, subscribed and paid-up capital: Rs. {inr(c['issued'])}.", s["Normal"]))
        P.add("authorised_capital", PDF_NAME, 4, auth_s)
    else:
        story.append(Paragraph(
            f"2. Share capital. Authorised: {auth_s} ({inr(shares_auth)} equity shares of Rs. 10 each), unchanged "
            f"from the previous year. Issued, subscribed and paid-up: Rs. {inr(c['issued'])}.", s["Normal"]))
        P.add("authorised_capital", PDF_NAME, 4, auth_s)
    story.append(Spacer(1, 8))

    bd, pbd = fmt_date(D(c["board"]), ds), fmt_date(D(c["prev_board"]), ds)
    story.append(Paragraph(
        f"3. Approval of financial statements. These financial statements were approved for issue by the Board "
        f"of Directors at its meeting held on {bd}. The financial statements for {plab} were approved by the Board "
        f"on {pbd}.", s["Normal"]))
    P.add("board_meeting_date", PDF_NAME, 4, bd)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"4. Revenue from operations for {ylab} is stated after eliminating inter-segment transfers. Comparative "
        f"figures for {plab} have been regrouped where necessary.", s["Normal"]))
    story.append(PageBreak())

    # pages 5-6: boilerplate notes (no target facts except the board date, which is recorded)
    story.append(Paragraph("5. Material accounting policies", s["Heading3"]))
    for para in ACCOUNTING_POLICIES:
        story.append(Paragraph(para, s["Normal"]))
        story.append(Spacer(1, 4))
    story.append(PageBreak())
    kmp = [("A. Samplewala", "Managing Director", 0.0021), ("B. Examplekar", "Chief Financial Officer", 0.0012),
           ("C. Placeholder", "Company Secretary", 0.0005)]
    story.append(Paragraph(f"6. Related party disclosures ({unit_note.strip('()')})", s["Heading3"]))
    rows = [["Key managerial personnel", "Designation", f"Remuneration {ylab}", f"Remuneration {plab}"]]
    for who, role, share in kmp:
        rows.append([who, role, u(_round_to(cur["pat"] * share, c)), u(_round_to(prev["pat"] * share, c))])
    story += [Table(rows, style=grid), Spacer(1, 8)]
    story.append(Paragraph(
        f"7. Contingent liabilities. Claims against the Company not acknowledged as debts amount to "
        f"{u(_round_to(cur['revenue'] * 0.004, c))} (previous year {u(_round_to(prev['revenue'] * 0.003, c))}). "
        f"Bank guarantees issued on behalf of the Company amount to {u(_round_to(cur['revenue'] * 0.007, c))}.",
        s["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"8. Auditors' remuneration. Statutory audit fee {u(_round_to(cur['revenue'] * 0.0003, c))}; tax audit fee "
        f"{u(_round_to(cur['revenue'] * 0.00005, c))}; other services {u(_round_to(cur['revenue'] * 0.00002, c))}.",
        s["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"9. Corporate social responsibility. The amount required to be spent under section 135 of the Act during "
        f"the year was {u(_round_to(prev['pbt'] * 0.02, c))} and the amount spent was "
        f"{u(_round_to(prev['pbt'] * 0.021, c))}, mainly on school education and drinking water projects.", s["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"10. Events after the reporting period. The Board of Directors, at its meeting held on {bd}, recommended a "
        f"final dividend of Rs. {c['agm_no'] % 4 + 1}.50 per equity share, subject to the approval of the members at "
        f"the ensuing Annual General Meeting. No other significant events have occurred after the balance sheet date.",
        s["Normal"]))
    P.add("board_meeting_date", PDF_NAME, 6, bd)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(40, 25, f"{FOOTER}  Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, title=f"{c['name']} financial statements (synthetic)",
                      author="Synthetic benchmark generator", invariant=1,
                      leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50).build(
        story, onFirstPage=footer, onLaterPages=footer)


def write_docx(c: dict, path: Path, P: Placements) -> None:
    import docx

    ds = c["docx_date_style"]
    d = docx.Document()
    cp = d.core_properties
    cp.author, cp.comments = "Synthetic benchmark generator", FOOTER
    cp.created = cp.modified = FIXED_TIME
    cp.last_modified_by = "Synthetic benchmark generator"
    cp.revision = 1
    fy_to_s = fmt_date(D(c["fy_to"]), ds)
    d.add_heading("Directors' Report", level=1)
    d.add_paragraph(FOOTER)
    d.add_paragraph(f"To the Members of {c['name']},")
    P.add("company_name", DOCX_NAME, None, c["name"])
    if c["fy_from_in"] == "docx":
        f = fmt_date(D(c["fy_from"]), ds)
        d.add_paragraph(f"Your Directors present the Annual Report of the Company together with the audited "
                        f"financial statements for the financial year {f} to {fy_to_s}.")
        P.add("fy_from", DOCX_NAME, None, f)
    else:
        d.add_paragraph(f"Your Directors present the Annual Report of the Company together with the audited "
                        f"financial statements for the financial year ended {fy_to_s}.")
    P.add("fy_to", DOCX_NAME, None, fy_to_s)

    d.add_heading("Financial highlights", level=2)
    d.add_paragraph(UNITS[c["unit"]][1])
    cur, prev = figures(c, False), figures(c, True)
    t = d.add_table(rows=1, cols=3)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = \
        "Particulars", fy_label(c), fy_label(c, prior=True)
    for label, key in [("Revenue from operations", "revenue"), ("Profit before tax", "pbt"),
                       ("Profit after tax", "pat")]:
        row = t.add_row().cells
        row[0].text, row[1].text, row[2].text = label, in_unit(cur[key], c["unit"]), in_unit(prev[key], c["unit"])
    P.add("revenue", DOCX_NAME, None, in_unit(cur["revenue"], c["unit"]))
    P.add("profit_after_tax", DOCX_NAME, None, in_unit(cur["pat"], c["unit"]))

    d.add_heading("State of the Company's affairs", level=2)
    d.add_paragraph(f"The Company continues to be engaged in {c['industry'][0].lower()}{c['industry'][1:]}. "
                    f"There was no change in the nature of business during the year.")
    P.add("industry", DOCX_NAME, None, f"{c['industry'][0].lower()}{c['industry'][1:]}")

    d.add_heading("Meetings of the Board", level=2)
    dates = [fmt_date(D(x), ds) for x in c["board_list"]]
    d.add_paragraph(f"During the year the Board met {len(dates)} times, on " + ", ".join(dates[:-1]) +
                    f" and {dates[-1]}. The meeting held on {fmt_date(D(c['prev_board']), ds)} approved the "
                    f"financial statements for the previous financial year.")

    d.add_heading("Subsidiaries and consolidated financial statements", level=2)
    d.add_paragraph(c["consol_text"])
    if c["holding"]:
        d.add_paragraph(f"Holding company: {c['holding']['name']}, CIN {c['holding']['cin']}.")

    d.add_heading("Secretarial audit", level=2)
    d.add_paragraph("Pursuant to section 204 of the Companies Act, 2013, the Board appointed A. Sample & Associates, "
                    "Practising Company Secretaries, to conduct the Secretarial Audit of the Company for the year. "
                    "The Secretarial Audit Report in Form MR-3 is annexed to this report as Annexure III.")
    if c.get("secretarial_distractor"):
        d.add_paragraph(c["secretarial_distractor"])

    if c["cag"]:
        d.add_heading("Comptroller and Auditor General of India", level=2)
        d.add_paragraph(c["cag"]["text"])

    d.add_heading("Annual General Meeting", level=2)
    prev_agm = fmt_date(D(c["prev_agm"]), ds)
    if c["agm"]:
        agm = D(c["agm"])
        agm_s = fmt_date(agm, ds)
        d.add_paragraph(f"The {ordinal(c['agm_no'])} Annual General Meeting of the Company will be held on "
                        f"{agm.strftime('%A')}, {agm_s} through video conferencing. The "
                        f"{ordinal(c['agm_no'] - 1)} Annual General Meeting was held on {prev_agm}.")
        P.add("agm_date", DOCX_NAME, None, agm_s)
    else:
        d.add_paragraph(f"The date, time and venue of the {ordinal(c['agm_no'])} Annual General Meeting will be "
                        f"communicated to the Members separately through the notice of the meeting. The "
                        f"{ordinal(c['agm_no'] - 1)} Annual General Meeting was held on {prev_agm}.")

    d.add_heading("Dividend", level=2)
    d.add_paragraph(f"The Board has recommended a final dividend of Rs. {c['agm_no'] % 4 + 1}.50 per equity share for "
                    f"the year, subject to the approval of the members.")
    d.add_heading("Committees of the Board", level=2)
    audit_dates = [fmt_date(D(x) + timedelta(days=1), ds) for x in c["board_list"]]
    d.add_paragraph(f"The Audit Committee met {len(audit_dates)} times during the year, on " + ", ".join(audit_dates[:-1])
                    + f" and {audit_dates[-1]}. The Corporate Social Responsibility Committee met on "
                      f"{fmt_date(D(c['board_list'][1]) + timedelta(days=3), ds)}.")
    d.add_heading("Statutory auditors", level=2)
    first_agm = D(c["prev_agm"]).replace(year=D(c["prev_agm"]).year - 2)
    d.add_paragraph(f"Example and Co., Chartered Accountants (Firm Registration No. 000000Z), were appointed as the "
                    f"statutory auditors of the Company at the {ordinal(c['agm_no'] - 3)} Annual General Meeting held "
                    f"on {fmt_date(first_agm, ds)} for a term of five consecutive years. Their report on the financial "
                    f"statements does not contain any qualification, reservation or adverse remark.")
    for heading, text in DOCX_BOILERPLATE:
        d.add_heading(heading, level=2)
        d.add_paragraph(text)

    d.add_heading("Investor contact", level=2)
    if "docx" in c["email_in"]:
        d.add_paragraph(f"Members may write to the Company Secretary at {c['email']} or to the Registrar and Share "
                        f"Transfer Agent at {c['rta_email']}.")
        P.add("email", DOCX_NAME, None, c["email"])
    else:
        d.add_paragraph(f"Members may send their queries to the Registrar and Share Transfer Agent at "
                        f"{c['rta_email']}.")
    d.add_paragraph(f"Registered office: {c['address']}")
    P.add("registered_office", DOCX_NAME, None, c["address"])

    bd = fmt_date(D(c["board"]), ds)
    d.add_paragraph(f"For and on behalf of the Board of Directors of {c['name']}")
    d.add_paragraph(f"Place: {c['address'].split(', ')[-1].rsplit(' ', 1)[0]}    Date: {bd}")
    P.add("board_meeting_date", DOCX_NAME, None, bd)
    d.save(path)


def write_xlsx(c: dict, path: Path, P: Placements) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wb.properties.creator = "Synthetic benchmark generator"
    wb.properties.created = wb.properties.modified = FIXED_TIME
    ws = wb.active
    ws.title = "Company master"
    ws.append(["Field", "Value"])
    ws.append(["Company name", c["name"]])
    ws.append(["CIN", c["cin"]])
    P.add("company_name", XLSX_NAME, None, c["name"])
    P.add("cin", XLSX_NAME, None, c["cin"])
    if c["fy_from_in"] == "xlsx":
        f, t = fmt_date(D(c["fy_from"]), c["date_style"]), fmt_date(D(c["fy_to"]), c["date_style"])
        ws.append(["Financial year", f"{f} to {t}"])
        P.add("fy_from", XLSX_NAME, None, f)
        P.add("fy_to", XLSX_NAME, None, t)
    else:
        ws.append(["Financial year", fy_label(c)])
    if "xlsx" in c["email_in"]:
        ws.append(["E-mail", c["email"]])
        P.add("email", XLSX_NAME, None, c["email"])
    ws.append(["Amounts in trial balance", "Rupees"])
    ws.append(["Note", FOOTER])

    for prior in (False, True):
        f = figures(c, prior)
        sheet = wb.create_sheet(f"TB {fy_label(c, prior).replace('FY ', 'FY')}")
        sheet.append(["Account", "Debit (Rs.)", "Credit (Rs.)"])
        opening_equity = f["other_equity"] - f["pat"]
        rows = [("Equity share capital", 0, f["capital"]), ("Other equity (opening)", 0, opening_equity),
                ("Borrowings", 0, f["borrowings"]), ("Trade payables", 0, f["payables"]),
                ("Property, plant and equipment", f["ppe"], 0), ("Inventories", f["inventories"], 0),
                ("Trade receivables", f["receivables"], 0), ("Cash and cash equivalents", f["cash"], 0),
                ("Revenue from operations", 0, f["revenue"]), ("Other income", 0, f["other_income"]),
                ("Operating and other expenses", f["expenses"], 0), ("Tax expense", f["tax"], 0)]
        for row in rows:
            sheet.append(list(row))
        dr, cr = sum(r[1] for r in rows), sum(r[2] for r in rows)
        assert dr == cr, (c["id"], prior, dr, cr)
        sheet.append(["Total", dr, cr])
        if not prior:
            P.add("revenue", XLSX_NAME, None, str(f["revenue"]))
    wb.save(path)


def normalise_zip(path: Path) -> None:
    """Rewrite an Office file with fixed timestamps (zip members and properties) so rebuilds are byte-identical."""
    with zipfile.ZipFile(path) as z:
        members = [(i.filename, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            if name == "docProps/core.xml":   # openpyxl stamps the save time into the properties
                data = re.sub(rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:)",
                              rb"\g<1>2026-01-01T00:00:00Z\g<3>", data)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)


# --------------------------------------------------------------------------- ground truth + checks


def gold_values(c: dict) -> dict:
    cag = c["cag"] or {"comments": False, "supplementary": False}
    yn = lambda b: "Yes" if b else "No"  # noqa: E731
    return {
        "cin": c["cin"], "company_name": c["name"], "registered_office": c["address"], "email": c["email"],
        "fy_from": c["fy_from"], "fy_to": c["fy_to"], "board_meeting_date": c["board"], "agm_date": c["agm"],
        "authorised_capital": str(c["auth"]), "industry": c["industry"],
        "consolidated_fs": yn(c["consolidated"]), "cag_comments": yn(cag["comments"]),
        "cag_supplementary_audit": yn(cag["supplementary"]), "secretarial_audit": "Yes",
        "revenue": str(c["revenue"]), "profit_after_tax": str(c["pat"]),
    }


# Yes/No answers are decided by sentences rather than a single value; these are their locations.
# For a non-government company the CAG answers follow from the note that says so (PDF page 4).
# Each entry is (file, page, a phrase from the sentence that decides the answer).
def yesno_sources(c: dict) -> dict:
    src = {"consolidated_fs": [(DOCX_NAME, None, c["consol_text"][:60].strip())],
           "secretarial_audit": [(DOCX_NAME, None, "The Secretarial Audit Report in Form MR-3")]}
    govt = c["cin"][12:15] in ("GOI", "SGC")
    cag = (([(DOCX_NAME, None, c["cag"]["text"][:60].strip())] if c["cag"] else []) +
           ([] if govt else [(PDF_NAME, 4, "not a Government company")]))
    src["cag_comments"] = src["cag_supplementary_audit"] = cag
    return src


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def extracted_text(folder: Path) -> dict:
    from openpyxl import load_workbook
    from pypdf import PdfReader

    out = {}
    for i, page in enumerate(PdfReader(str(folder / PDF_NAME)).pages, start=1):
        out[(PDF_NAME, i)] = _squash(page.extract_text() or "")
    out[(DOCX_NAME, None)] = _squash(docx_text(folder / DOCX_NAME))
    wb = load_workbook(str(folder / XLSX_NAME), data_only=True)
    out[(XLSX_NAME, None)] = _squash(" ".join(str(v) for ws in wb.worksheets for row in ws.iter_rows(values_only=True)
                                              for v in row if v is not None))
    return out


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {"description": "Synthetic AOC-4 style field-extraction benchmark. All companies and figures are "
                            "fictional. Each field lists its gold value (null = not stated in the documents) and "
                            "the (file, page) locations that state it, with the text that states it there "
                            "(evidence); page is null for .docx and .xlsx files.",
             "fields": FIELDS, "companies": []}
    for c in COMPANIES:
        folder = out_dir / c["id"]
        folder.mkdir(exist_ok=True)
        P = Placements()
        write_pdf(c, folder / PDF_NAME, P)
        write_docx(c, folder / DOCX_NAME, P)
        write_xlsx(c, folder / XLSX_NAME, P)
        normalise_zip(folder / DOCX_NAME)
        normalise_zip(folder / XLSX_NAME)

        gold = gold_values(c)
        ys = yesno_sources(c)
        texts = extracted_text(folder)
        for field, places in list(P.by_field.items()) + list(ys.items()):
            for file, page, surface in places:
                assert _squash(surface) in texts[(file, page)], (c["id"], field, file, page, surface)

        fields = {}
        for f in FIELDS:
            fid = f["id"]
            places = [] if gold[fid] is None else ys.get(fid) or P.by_field.get(fid, [])
            assert gold[fid] is None or places, (c["id"], fid)
            sources = []
            for file, page, surface in places:
                entry = next((s for s in sources if s["file"] == file and s["page"] == page), None)
                if entry is None:
                    entry = {"file": file, "page": page, "evidence": []}
                    sources.append(entry)
                if surface not in entry["evidence"]:
                    entry["evidence"].append(surface)
            fields[fid] = {"gold": gold[fid], "sources": sources}
        truth["companies"].append({"id": c["id"], "folder": c["id"], "name": c["name"],
                                   "amount_unit_in_statements": c["unit"], "fields": fields})
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "data" / "dam"), help="output folder (default: benchmarks/data/dam)")
    args = ap.parse_args()
    truth = build(Path(args.out))
    n = sum(len(c["fields"]) for c in truth["companies"])
    nulls = sum(v["gold"] is None for c in truth["companies"] for v in c["fields"].values())
    print(f"{len(truth['companies'])} companies, {n} field instances ({nulls} not stated) -> {args.out}")


if __name__ == "__main__":
    main()
