"""AOC-4 style fields used by the DAM field-extraction benchmark.

Each field has an id, the form label used as the retrieval query and question
(modelled on the labels in DAM/visual_grounding.py), a value type that decides
how answers are normalised and scored, and a one-line format hint that every
LLM-based method receives.
"""

FIELD_TYPES = ("id", "text", "email", "date", "amount", "yesno")

FORMAT_HINTS = {
    "id": "the identifier exactly as written",
    "text": "the text exactly as written",
    "email": "the e-mail address",
    "date": "a date in DD/MM/YYYY format",
    "amount": "an amount in rupees written as a whole number (convert lakhs or crores to rupees)",
    "yesno": "Yes or No",
}

FIELDS = [
    {"id": "cin", "type": "id",
     "label": "1.(a) Corporate Identity Number (CIN) of company"},
    {"id": "company_name", "type": "text",
     "label": "2.(a) Name of the company"},
    {"id": "registered_office", "type": "text",
     "label": "2.(b) Address of the registered office of the company"},
    {"id": "email", "type": "email",
     "label": "2.(c) e-mail ID of the company"},
    {"id": "fy_from", "type": "date",
     "label": "3. Financial year to which the financial statements relate - From (DD/MM/YYYY)"},
    {"id": "fy_to", "type": "date",
     "label": "3. Financial year to which the financial statements relate - To (DD/MM/YYYY)"},
    {"id": "board_meeting_date", "type": "date",
     "label": "4.(a) Date of Board of Directors' meeting in which financial statements are approved (DD/MM/YYYY)"},
    {"id": "agm_date", "type": "date",
     "label": "5.(b) Date of the Annual General Meeting (AGM) for this financial year (DD/MM/YYYY)"},
    {"id": "authorised_capital", "type": "amount",
     "label": "Authorised capital of the company at the end of the financial year (in Rs.)"},
    {"id": "industry", "type": "text",
     "label": "7. Type of industry / principal business activity of the company"},
    {"id": "consolidated_fs", "type": "yesno",
     "label": "8. Whether consolidated financial statements are also being filed (Yes/No)"},
    {"id": "cag_comments", "type": "yesno",
     "label": "9.(a) In case of a government company, whether the CAG of India has commented upon or "
              "supplemented the audit report under section 143 of the Companies Act, 2013 (Yes/No)"},
    {"id": "cag_supplementary_audit", "type": "yesno",
     "label": "9.(d) Whether the CAG of India has conducted a supplementary or test audit under "
              "section 143 (Yes/No)"},
    {"id": "secretarial_audit", "type": "yesno",
     "label": "10. Whether Secretarial Audit is applicable to the company (Yes/No)"},
    {"id": "revenue", "type": "amount",
     "label": "Revenue from operations of the company for the current financial year (in Rs.)"},
    {"id": "profit_after_tax", "type": "amount",
     "label": "Profit after tax of the company for the current financial year (in Rs.)"},
]

FIELD_BY_ID = {f["id"]: f for f in FIELDS}


def question(field: dict) -> str:
    """The question text an LLM-based method sees for one field."""
    return f"{field['label']}\nAnswer format: {FORMAT_HINTS[field['type']]}."
