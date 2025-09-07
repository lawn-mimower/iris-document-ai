import fitz  # PyMuPDF
import cv2
import numpy as np
import json
import argparse
import os

# --- Configuration ---
# Define colors for different element types for clear visualization
COLORS = {
    "TEXT": (255, 0, 0),        # Blue for text
    "INPUT_FIELD": (0, 255, 0), # Green for input fields
    "CHECKBOX": (0, 0, 255),      # Red for checkboxes
    "DEFAULT": (0, 255, 255)      # Yellow for any other type
}
DPI = 300 # Render DPI, should match the perception tool for accuracy

# --- IRIS Project Paths ---
DEFAULT_PDF_PATH = "phase1/input/Claim_Form.pdf"
DEFAULT_RAW_JSON_PATH = "phase1/output/Claim_Form_elements.json"
DEFAULT_PROCESSED_JSON_PATH = "processed_output.json"
DEFAULT_OUTPUT_DIR = "visualizations"

def draw_bboxes(pdf_path, json_path, output_path, page_num_to_draw=0):
    """
    Draws bounding boxes from a JSON file onto an image of a PDF page.
    """
    # 1. --- Load the JSON data ---
    try:
        with open(json_path, 'r') as f:
            elements = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    # Filter elements for the specific page we want to visualize
    page_elements = [el for el in elements if el.get('page_num') == page_num_to_draw]
    if not page_elements:
        print(f"No elements found for page {page_num_to_draw} in the JSON file.")
        return

    # 2. --- Render the specified PDF page to an image ---
    try:
        doc = fitz.open(pdf_path)
        if page_num_to_draw >= len(doc):
            print(f"Error: Page {page_num_to_draw} does not exist in the PDF.")
            doc.close()
            return
        page = doc[page_num_to_draw]
        
        # Get page dimensions before creating pixmap
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Create a pixmap (image rendering) of the page
        pix = page.get_pixmap(dpi=DPI)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert from RGB (PyMuPDF) to BGR (OpenCV)
        img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        # Calculate scale factors
        scale_x = pix.w / page_width
        scale_y = pix.h / page_height

    except Exception as e:
        print(f"Error processing PDF file: {e}")
        return
    finally:
        if 'doc' in locals():
            doc.close()

    # 3. --- Draw each bounding box on the image ---
    for el in page_elements:
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
            
        # The bbox coordinates are in PDF "points". We need to scale them
        # to the pixel coordinates of our rendered image.

        x0 = int(bbox[0] * scale_x)
        y0 = int(bbox[1] * scale_y)
        x1 = int(bbox[2] * scale_x)
        y1 = int(bbox[3] * scale_y)

        # Get the color for the element type, with a default
        color = COLORS.get(el.get("type"), COLORS["DEFAULT"])
        
        # Draw the rectangle on the image
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), color, 2) # 2 is the line thickness

    # 4. --- Save the final image ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv2.imwrite(output_path, img_cv)
    print(f"Successfully created visualization at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize element bounding boxes on a PDF page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize raw elements from Phase 1
  python visualise_elements.py

  # Visualize processed elements from gestalt_processor.py
  python visualise_elements.py --processed

  # Custom paths
  python visualise_elements.py my.pdf my_elements.json output.png

  # Specific page
  python visualise_elements.py --page 1
        """
    )
    
    parser.add_argument("pdf_path", type=str, nargs='?', default=DEFAULT_PDF_PATH, 
                       help=f"Path to the source PDF file (default: {DEFAULT_PDF_PATH})")
    parser.add_argument("json_path", type=str, nargs='?', default=DEFAULT_RAW_JSON_PATH,
                       help=f"Path to the JSON file with element data (default: {DEFAULT_RAW_JSON_PATH})")
    parser.add_argument("output_path", type=str, nargs='?', 
                       help="Path to save the output image file (default: auto-generated)")
    parser.add_argument("--page", type=int, default=0, 
                       help="The page number of the PDF to visualize (0-indexed, default: 0)")
    parser.add_argument("--processed", action="store_true", 
                       help="Use processed JSON from gestalt_processor.py instead of raw JSON")
    
    args = parser.parse_args()
    
    # Handle --processed flag
    if args.processed and args.json_path == DEFAULT_RAW_JSON_PATH:
        args.json_path = DEFAULT_PROCESSED_JSON_PATH
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        json_type = "processed" if args.processed or "processed" in args.json_path else "raw"
        args.output_path = f"{DEFAULT_OUTPUT_DIR}/elements_{json_type}_page_{args.page}.png"

    draw_bboxes(args.pdf_path, args.json_path, args.output_path, args.page)


if __name__ == "__main__":
    main()
