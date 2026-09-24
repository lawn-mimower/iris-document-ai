#!/usr/bin/env python3
"""
Intelligent Inventory Management System MVP
Parses Bill of Sale text and loads data into Neo4j knowledge graph
"""

import re
import os
from neo4j import GraphDatabase
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_bill_of_sale(text: str) -> Dict[str, Any]:
    """
    Parse a text-based Bill of Sale to extract key entities and tabular data.
    
    Args:
        text (str): Raw text from the bill of sale
        
    Returns:
        Dict containing seller, buyer, invoice_id, date, and items list
    """
    lines = text.strip().split('\n')
    
    # Initialize result dictionary
    result = {
        'seller': '',
        'buyer': '',
        'invoice_id': '',
        'date': '',
        'items': []
    }
    
    # Find seller information (after "Bill From:")
    bill_from_found = False
    for i, line in enumerate(lines):
        if "Bill From:" in line:
            bill_from_found = True
            # Get the company name from the next non-empty line
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and "Bill To:" not in lines[j]:
                    result['seller'] = lines[j].strip()
                    break
            break
    
    # Find buyer information (after "Bill To:")
    bill_to_found = False
    for i, line in enumerate(lines):
        if "Bill To:" in line:
            bill_to_found = True
            # Get the company name from the next non-empty line
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and "Invoice #:" not in lines[j]:
                    result['buyer'] = lines[j].strip()
                    break
            break
    
    # Find invoice ID
    for line in lines:
        if "Invoice #:" in line:
            result['invoice_id'] = line.split("Invoice #:")[1].strip()
            break
    
    # Find date
    for line in lines:
        if "Date:" in line:
            result['date'] = line.split("Date:")[1].strip()
            break
    
    # Find and parse the items table
    table_start = -1
    for i, line in enumerate(lines):
        if "Item" in line and "Quantity" in line and "Unit Price" in line and "Total" in line:
            table_start = i
            break
    
    if table_start != -1:
        # Skip the header and separator lines
        for i in range(table_start + 2, len(lines)):
            line = lines[i].strip()
            
            # Stop when we reach subtotal
            if "Subtotal:" in line or "Tax" in line or "Total:" in line:
                break
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse table row using | as delimiter
            if "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 3:
                    item_name = parts[0]
                    quantity_str = parts[1]
                    unit_price_str = parts[2]
                    
                    # Extract numeric values
                    try:
                        quantity = int(quantity_str)
                        # Remove $ and convert to float
                        unit_price = float(unit_price_str.replace('$', '').replace(',', ''))
                        
                        result['items'].append({
                            'name': item_name,
                            'quantity': quantity,
                            'unit_price': unit_price
                        })
                    except ValueError:
                        # Skip lines that can't be parsed as numbers
                        continue
    
    return result


def load_data_to_neo4j(data: Dict[str, Any], uri: str, user: str, password: str) -> None:
    """
    Load parsed bill of sale data into Neo4j knowledge graph.
    
    Args:
        data: Dictionary containing parsed bill of sale data
        uri: Neo4j database URI
        user: Neo4j username
        password: Neo4j password
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Create or find seller company
            session.run("""
                MERGE (seller:Company {name: $seller_name})
                SET seller.type = 'Seller'
            """, seller_name=data['seller'])
            
            # Create or find buyer company
            session.run("""
                MERGE (buyer:Company {name: $buyer_name})
                SET buyer.type = 'Buyer'
            """, buyer_name=data['buyer'])
            
            # Create invoice node (MERGE keeps re-runs from duplicating the invoice)
            session.run("""
                MERGE (invoice:Invoice {id: $invoice_id})
                SET invoice.date = $date
            """, invoice_id=data['invoice_id'], date=data['date'])
            
            # Create relationships between companies and invoice
            session.run("""
                MATCH (seller:Company {name: $seller_name})
                MATCH (invoice:Invoice {id: $invoice_id})
                MERGE (seller)-[:SOLD]->(invoice)
            """, seller_name=data['seller'], invoice_id=data['invoice_id'])
            
            session.run("""
                MATCH (buyer:Company {name: $buyer_name})
                MATCH (invoice:Invoice {id: $invoice_id})
                MERGE (buyer)-[:BOUGHT]->(invoice)
            """, buyer_name=data['buyer'], invoice_id=data['invoice_id'])
            
            # Process each item
            for item in data['items']:
                # Create or find product
                session.run("""
                    MERGE (product:Product {name: $product_name})
                """, product_name=item['name'])
                
                # Create relationship with quantity and unit_price properties
                session.run("""
                    MATCH (invoice:Invoice {id: $invoice_id})
                    MATCH (product:Product {name: $product_name})
                    MERGE (invoice)-[item:CONTAINS_ITEM]->(product)
                    SET item.quantity = $quantity,
                        item.unit_price = $unit_price
                """, 
                invoice_id=data['invoice_id'],
                product_name=item['name'],
                quantity=item['quantity'],
                unit_price=item['unit_price'])
            
            print(f"Successfully loaded invoice {data['invoice_id']} to Neo4j database")
            
    except Exception as e:
        print(f"Error loading data to Neo4j: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    # Sample Bill of Sale text
    sample_bill_of_sale_text = """
INVOICE

Bill From:
SupplierCorp Inc.
123 Industrial Way
Metropolis, 12345

Bill To:
MyCompany LLC
456 Business Blvd
Gotham, 67890

Invoice #: INV-2025-001
Date: 2025-09-18

Item             | Quantity | Unit Price | Total
-----------------|----------|------------|--------
Part #XYZ-100    | 150      | $25.50     | $3825.00
Widget Model A   | 200      | $12.00     | $2400.00
Assembly Kit B   | 50       | $150.00    | $7500.00

Subtotal: $13725.00
Tax (5%): $686.25
Total: $14411.25
"""
    
    # Parse the bill of sale
    print("Parsing bill of sale...")
    parsed_data = parse_bill_of_sale(sample_bill_of_sale_text)
    print("Parsed data:")
    print(f"  Seller: {parsed_data['seller']}")
    print(f"  Buyer: {parsed_data['buyer']}")
    print(f"  Invoice ID: {parsed_data['invoice_id']}")
    print(f"  Date: {parsed_data['date']}")
    print(f"  Items: {len(parsed_data['items'])} products")
    for item in parsed_data['items']:
        print(f"    - {item['name']}: {item['quantity']} @ ${item['unit_price']}")
    
    # Neo4j connection details from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Load data to Neo4j
    try:
        print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
        load_data_to_neo4j(parsed_data, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("✅ Invoice data successfully loaded into Neo4j knowledge graph!")
    except Exception as e:
        print(f"❌ Failed to load data to Neo4j: {e}")
        print("Please ensure Neo4j is running and credentials are correct.")