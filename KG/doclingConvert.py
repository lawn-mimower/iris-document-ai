import argparse
from docling.document_converter import DocumentConverter

# 1. Create a parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Convert a document to Markdown using docling.")

# 2. Add an argument for the source file/URL (required positional)
parser.add_argument("source", help="The local path or URL of the document to convert.")

# 3. Add an optional argument for the output Markdown file
parser.add_argument("-o", "--output", default="output.md", help="Path to save the converted Markdown file (default: output.md)")

# 4. Parse the arguments provided in the terminal
args = parser.parse_args()

# 5. Use the 'source' argument from the command line
source = args.source
output_path = args.output

converter = DocumentConverter()
result = converter.convert(source)

# 6. Write the output to a file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())
