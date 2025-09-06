import fitz  # PyMuPDF
import cv2
import numpy as np
import json
import os
import argparse
import pytesseract

# --- Configuration ---
# If tesseract is not in your PATH, uncomment and set the path below
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DPI = 300  # High DPI for accurate CV analysis
IOU_THRESHOLD = 0.4  # Intersection over Union threshold for associating elements

def calculate_iou(box_a, box_b):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box_a (list): [x1, y1, x2, y2]
        box_b (list): [x1, y1, x2, y2]
    Returns:
        float: The IoU value, between 0 and 1.
    """
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    # Compute the area of both bounding boxes
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the IoU
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def detect_indicator_shapes(image, scale_x=1.0, scale_y=1.0):
    """
    Detects shapes like underlines and boxes that indicate input fields.
    This is a specialized replacement for the old generic shape detector.
    """
    indicator_shapes = []
    
    # Pre-process image for shape detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better results on varied lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- Heuristics to find Underlines and Boxes ---
        aspect_ratio = w / h if h > 0 else 0
        
        # Heuristic for underlines: very wide and short
        is_underline = aspect_ratio > 10 and h < 10

        # Heuristic for boxes: reasonable size, not a line
        is_box = (w > 15 and h > 15) and (0.5 < aspect_ratio < 10)

        if is_underline or is_box:
            # Scale coordinates back to the original document space
            x0, y0 = x * scale_x, y * scale_y
            x1, y1 = (x + w) * scale_x, (y + h) * scale_y
            
            shape = {
                "type": "UNDERLINE" if is_underline else "BOX",
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
            }
            indicator_shapes.append(shape)
            
    return indicator_shapes

def process_pdf(pdf_path):
    """
    Revised function to extract elements using the overlap/re-classification logic.
    """
    doc = fitz.open(pdf_path)
    all_elements = []
    
    for page_num, page in enumerate(doc):
        # 1. --- Extract all raw text elements first ---
        candidate_elements = []
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Filter out purely whitespace spans
                        if not span["text"].strip():
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        element = {
                            "type": "TEXT",
                            "text": span["text"],
                            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                            "center": [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)],
                            "font_size": round(span["size"]),
                            "font_weight": "bold" if "bold" in span["font"].lower() else "normal",
                            "page_num": page_num
                        }
                        candidate_elements.append(element)

        # 2. --- Detect all indicator shapes (underlines and boxes) ---
        pix = page.get_pixmap(dpi=DPI)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        scale_x = page.rect.width / pix.w
        scale_y = page.rect.height / pix.h
        indicator_shapes = detect_indicator_shapes(img_cv, scale_x, scale_y)
        
        # 3. --- Correlate and Re-classify based on Overlap ---
        final_elements = []
        used_indicator_indices = set()

        for el in candidate_elements:
            is_input_field = False
            for i, shape in enumerate(indicator_shapes):
                if i in used_indicator_indices:
                    continue
                iou = calculate_iou(el["bbox"], shape["bbox"])
                if iou > IOU_THRESHOLD:
                    # High overlap with an indicator means this is an input field
                    is_input_field = True
                    used_indicator_indices.add(i)
                    break # Associate with the first indicator found

            if is_input_field:
                el["type"] = "INPUT_FIELD"
                # Often, text like '____' is read; we can clear it
                if all(c in ' _' for c in el["text"]):
                    el["text"] = None
            final_elements.append(el)

        # Add any indicator boxes that did NOT overlap with text as new input fields
        for i, shape in enumerate(indicator_shapes):
            if i not in used_indicator_indices and shape["type"] == "BOX":
                new_field = {
                    "type": "INPUT_FIELD",
                    "text": None,
                    "bbox": shape["bbox"],
                    "center": [round((shape["bbox"][0] + shape["bbox"][2]) / 2, 2),
                               round((shape["bbox"][1] + shape["bbox"][3]) / 2, 2)],
                    "page_num": page_num
                }
                final_elements.append(new_field)

        all_elements.extend(final_elements)

    doc.close()
    return all_elements

# The process_image and main functions would be updated similarly but are omitted here for brevity.
# The core logic change is demonstrated in process_pdf. We can assume a similar update for process_image.

def main():
    parser = argparse.ArgumentParser(description="IRIS Perception Tool v2: Extract elements using shape correlation.")
    parser.add_argument("input_path", type=str, help="Path to the input PDF or image file.")
    parser.add_argument("output_path", type=str, help="Path to save the output JSON file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at {args.input_path}")
        return

    print(f"Processing document: {args.input_path}")
    
    file_ext = os.path.splitext(args.input_path)[1].lower()
    elements = []

    if file_ext == ".pdf":
        elements = process_pdf(args.input_path)
    # The process_image function would need a similar overhaul to use this logic
    # elif file_ext in [".png", ".jpg", ".jpeg", ".tiff"]:
    #     elements = process_image(args.input_path) 
    else:
        print(f"Error: Unsupported or logic not yet implemented for '{file_ext}'. Please use PDF.")
        return

    # Add a unique ID to each element
    for i, el in enumerate(elements):
        el["id"] = f"element_{i}"

    print(f"Successfully extracted {len(elements)} elements.")

    with open(args.output_path, "w") as f:
        json.dump(elements, f, indent=2)
    
    print(f"Output saved to {args.output_path}")

if __name__ == "__main__":
    main()