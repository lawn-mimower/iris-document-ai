import pandas as pd
from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Convert document

result = converter.convert("policy.pdf")

# Create output directory
output_dir = Path("extracted_tables")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract and save tables
for table_ix, table in enumerate(result.document.tables):
    # Export to pandas DataFrame
    table_df = table.export_to_dataframe()
    
    # Display as Markdown
    print(f"## Table {table_ix + 1}")
    print(table_df.to_markdown())
    
    # Save as CSV
    csv_filename = output_dir / f"table_{table_ix + 1}.csv"
    table_df.to_csv(csv_filename)
    
    # Save as HTML
    html_filename = output_dir / f"table_{table_ix + 1}.html"
    with html_filename.open("w") as f:
        f.write(table.export_to_html(doc=result.document))
