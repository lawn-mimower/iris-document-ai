"""KG/inventory_management.py: bill-of-sale parsing and Neo4j loading."""

import os

import pytest

from conftest import load_module

inv = load_module("KG/inventory_management.py", "inventory_management")

BILL = """
INVOICE

Bill From:
Demo Parts Supplier (fictional)
1 Sample Road

Bill To:
Example Buyer Co (fictional)
2 Placeholder Street

Invoice #: INV-TEST-0001
Date: 2025-01-15

Item             | Quantity | Unit Price | Total
-----------------|----------|------------|--------
Test Bolt M6     | 10       | $1,250.50  | $12505.00
Test Washer      | 5        | $2.00      | $10.00
Not a number row | many     | $3.00      | $3.00

Subtotal: $12515.00
Tax (5%): $625.75
Total: $13140.75
"""


def test_parse_bill_of_sale():
    data = inv.parse_bill_of_sale(BILL)
    assert data["seller"] == "Demo Parts Supplier (fictional)"
    assert data["buyer"] == "Example Buyer Co (fictional)"
    assert data["invoice_id"] == "INV-TEST-0001"
    assert data["date"] == "2025-01-15"
    assert data["items"] == [
        {"name": "Test Bolt M6", "quantity": 10, "unit_price": 1250.50},
        {"name": "Test Washer", "quantity": 5, "unit_price": 2.0},
    ]


def test_parse_without_table_returns_no_items():
    data = inv.parse_bill_of_sale("Bill From:\nA\nBill To:\nB\nInvoice #: X1\nDate: today")
    assert data["items"] == [] and data["invoice_id"] == "X1"


class FakeSession:
    def __init__(self, log):
        self.log = log

    def run(self, query, **params):
        self.log.append((" ".join(query.split()), params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self):
        self.log, self.closed = [], False

    def session(self):
        return FakeSession(self.log)

    def close(self):
        self.closed = True


def test_load_uses_merge_so_reruns_are_idempotent(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(inv.GraphDatabase, "driver", lambda uri, auth: driver)
    inv.load_data_to_neo4j(inv.parse_bill_of_sale(BILL), "bolt://x", "u", "p")
    queries = [q for q, _ in driver.log]
    assert driver.closed
    assert not any(q.startswith("CREATE") or " CREATE " in q for q in queries)
    assert any("MERGE (invoice:Invoice {id: $invoice_id})" in q for q in queries)
    item_params = [p for q, p in driver.log if "CONTAINS_ITEM" in q]
    assert [p["product_name"] for p in item_params] == ["Test Bolt M6", "Test Washer"]


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"),
                    reason="NEO4J_URI / NEO4J_PASSWORD not set")
def test_live_load_twice_into_neo4j():
    uri, user, pw = os.environ["NEO4J_URI"], os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]
    data = inv.parse_bill_of_sale(BILL)
    driver = inv.GraphDatabase.driver(uri, auth=(user, pw))

    def cleanup():
        with driver.session() as s:
            s.run("MATCH (i:Invoice {id: $id}) DETACH DELETE i", id=data["invoice_id"])
            s.run("MATCH (p:Product) WHERE p.name IN $names AND NOT (p)--() DELETE p",
                  names=[i["name"] for i in data["items"]])
            s.run("MATCH (c:Company) WHERE c.name IN $names AND NOT (c)--() DELETE c",
                  names=[data["seller"], data["buyer"]])

    try:
        cleanup()
        inv.load_data_to_neo4j(data, uri, user, pw)
        inv.load_data_to_neo4j(data, uri, user, pw)
        with driver.session() as s:
            rec = s.run(
                "MATCH (i:Invoice {id: $id}) "
                "OPTIONAL MATCH (i)-[r:CONTAINS_ITEM]->(:Product) "
                "RETURN count(DISTINCT i) AS invoices, count(r) AS items",
                id=data["invoice_id"],
            ).single()
        assert rec["invoices"] == 1 and rec["items"] == 2
    finally:
        cleanup()
        driver.close()
