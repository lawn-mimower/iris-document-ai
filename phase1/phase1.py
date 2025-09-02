import fitz  # PyMuPDF
import cv2
import numpy as np
import json
import os
import argparse
import pytesseract

# --- Configuration ---
# If tesseract is not in your PATH, include the following line with the path to your Tesseract installation
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_visual_elements(image, scale_x=1.0, scale_y=1.0):
    """
    Finds rectangular shapes in an image that are likely input fields or checkboxes.
    Args:
        image (np.array): The image to process (in OpenCV format).
        scale_x (float): Horizontal scale factor to convert image coords to PDF coords.
        scale_y (float): Vertical scale factor to convert image coords to PDF coords.
    Returns:
        list: A list of dictionaries, each representing a visual element.
    """
    visual_elements = []
    
    # Convert image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert and threshold the image to make form lines and boxes white
    gray_inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray_inv, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours (the outlines of shapes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- Filtering Heuristics ---
        # Filter for shapes that resemble input fields or checkboxes.
        # These values may need to be tuned for your specific forms.
        is_reasonable_size = (w > 10 and h > 10) and (w < image.shape[1] * 0.8)
        aspect_ratio = w / h if h > 0 else 0
        is_not_a_line = not (aspect_ratio > 10 or aspect_ratio < 0.1)

        if is_reasonable_size and is_not_a_line:
            # Scale coordinates back to the original document space (e.g., PDF points)
            x0, y0 = x * scale_x, y * scale_y
            x1, y1 = (x + w) * scale_x, (y + h) * scale_y
            
            # Classify as checkbox or field based on aspect ratio
            element_type = "CHECKBOX" if 0.8 < aspect_ratio < 1.2 and w < 50 else "INPUT_FIELD"
            
            element = {
                "type": element_type,
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "center": [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)],
            }
            visual_elements.append(element)
            
    return visual_elements

def process_pdf(pdf_path):
    """
    Extracts text and visual elements from a PDF file.
    """
    doc = fitz.open(pdf_path)
    all_elements = []
    
    for page_num, page in enumerate(doc):
        # 1. --- TEXT EXTRACTION (from PDF data) ---
        # This is highly accurate for digitally created PDFs.
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        element = {
                            "type": "TEXT",
                            "text": span["text"],
                            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                            "center": [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)],
                            "font_size": round(span["size"]),
                            "font_weight": "bold" if "bold" in span["font"].lower() else "normal"
                        }
                        all_elements.append(element)

        # 2. --- VISUAL EXTRACTION (from rendered image) ---
        # Render the page to an image to find visual shapes like boxes.
        pix = page.get_pixmap(dpi=300)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # PyMuPDF outputs RGB, OpenCV uses BGR
        img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        # Calculate scale factors to map image coordinates back to PDF coordinates
        scale_x = page.rect.width / pix.w
        scale_y = page.rect.height / pix.h
        
        visual_elements = extract_visual_elements(img_cv, scale_x, scale_y)
        for el in visual_elements:
            el["page_num"] = page_num
        all_elements.extend(visual_elements)

    doc.close()
    return all_elements

def process_image(image_path):
    """
    Extracts text (via OCR) and visual elements from an image file.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return []
        
    all_elements = []
    page_num = 0

    # 1. --- TEXT EXTRACTION (OCR) ---
    # Use pytesseract to get detailed data about words and their locations
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 60: # Filter out low-confidence detections
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            x0, y0, x1, y1 = x, y, x + w, y + h
            element = {
                "type": "TEXT",
                "text": ocr_data['text'][i],
                "bbox": [x0, y0, x1, y1],
                "center": [(x0 + x1) / 2, (y0 + y1) / 2],
                "font_size": None, # OCR doesn't typically provide this
                "font_weight": "normal"
            }
            all_elements.append(element)

    # 2. --- VISUAL EXTRACTION ---
    visual_elements = extract_visual_elements(img) # Scale is 1.0 for images
    for el in visual_elements:
        el["page_num"] = page_num
    all_elements.extend(visual_elements)
    
    return all_elements


def main():
    """
    Main function to parse command-line arguments and run the extraction process.
    """
    parser = argparse.ArgumentParser(description="IRIS Phase 1: Extract visual and textual elements from a document.")
    parser.add_argument("input_path", type=str, help="Path to the input PDF or image file.")
    parser.add_argument("output_path", type=str, help="Path to save the output JSON file.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Processing document: {input_path}")
    
    file_ext = os.path.splitext(input_path)[1].lower()
    elements = []

    if file_ext == ".pdf":
        elements = process_pdf(input_path)
    elif file_ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        elements = process_image(input_path)
    else:
        print(f"Error: Unsupported file type '{file_ext}'. Please use PDF or a standard image format.")
        return

    # Add a unique ID to each element for later reference
    for i, el in enumerate(elements):
        el["id"] = f"element_{i}"
        el.setdefault("page_num", 0) # Ensure page_num is set

    print(f"Successfully extracted {len(elements)} elements.")

    # Save the output to a JSON file
    with open(output_path, "w") as f:
        json.dump(elements, f, indent=2)
    
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
