#!/usr/bin/env python3
"""
IRIS Phase 1.5: Gestalt Pre-processor

This script processes raw JSON output from Phase 1 perception layer,
merging fragmented elements and enriching them with spatial relationship data
for use by the Phase 2 LLM agent.
"""

import json
import argparse
import math
from typing import List, Dict, Any, Tuple, Optional


def load_elements(input_file: str) -> List[Dict[str, Any]]:
    """
    Load elements from the input JSON file.
    
    Args:
        input_file: Path to the input JSON file
        
    Returns:
        List of element dictionaries
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            elements = json.load(f)
        
        if not isinstance(elements, list):
            raise ValueError("Input JSON must contain a list of elements")
            
        return elements
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in input file: {e}")


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the horizontal overlap between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        Horizontal overlap distance (0 if no overlap)
    """
    x1_left, _, x1_right, _ = bbox1
    x2_left, _, x2_right, _ = bbox2
    
    overlap = min(x1_right, x2_right) - max(x1_left, x2_left)
    return max(0, overlap)


def is_horizontally_adjacent(elem1: Dict[str, Any], elem2: Dict[str, Any], 
                           max_gap: float = 10.0, max_y_diff: float = 5.0) -> bool:
    """
    Check if two elements are horizontally adjacent and vertically aligned.
    
    Args:
        elem1, elem2: Element dictionaries with bbox and center coordinates
        max_gap: Maximum horizontal gap to consider adjacent
        max_y_diff: Maximum vertical difference to consider aligned
        
    Returns:
        True if elements should be merged horizontally
    """
    # Must be on same page
    if elem1.get('page_num') != elem2.get('page_num'):
        return False
    
    bbox1, bbox2 = elem1['bbox'], elem2['bbox']
    center1, center2 = elem1['center'], elem2['center']
    
    # Check vertical alignment (similar Y coordinates)
    y_diff = abs(center1[1] - center2[1])
    if y_diff > max_y_diff:
        return False
    
    # Check horizontal adjacency
    x1_left, y1_top, x1_right, y1_bottom = bbox1
    x2_left, y2_top, x2_right, y2_bottom = bbox2
    
    # Calculate gap between elements
    if x1_right <= x2_left:  # elem1 is to the left of elem2
        gap = x2_left - x1_right
    elif x2_right <= x1_left:  # elem2 is to the left of elem1
        gap = x1_left - x2_right
    else:  # Elements overlap horizontally
        gap = 0
    
    return gap <= max_gap


def merge_text_fragments(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge horizontally adjacent TEXT elements into single elements.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        List with merged text elements
    """
    text_elements = [elem for elem in elements if elem.get('type') == 'TEXT']
    non_text_elements = [elem for elem in elements if elem.get('type') != 'TEXT']
    
    merged_groups = []
    used_indices = set()
    
    for i, elem in enumerate(text_elements):
        if i in used_indices:
            continue
            
        # Start a new merge group
        merge_group = [elem]
        used_indices.add(i)
        
        # Find all elements that should be merged with this one
        changed = True
        while changed:
            changed = False
            for j, other_elem in enumerate(text_elements):
                if j in used_indices:
                    continue
                    
                # Check if this element is adjacent to any element in current group
                for group_elem in merge_group:
                    if is_horizontally_adjacent(group_elem, other_elem):
                        merge_group.append(other_elem)
                        used_indices.add(j)
                        changed = True
                        break
        
        merged_groups.append(merge_group)
    
    # Create merged elements
    merged_elements = []
    for group_idx, group in enumerate(merged_groups):
        if len(group) == 1:
            # Single element, keep as is
            merged_elements.append(group[0])
        else:
            # Merge multiple elements
            # Sort by x-coordinate to maintain text order
            group.sort(key=lambda x: x['bbox'][0])
            
            merged_text = ' '.join(elem.get('text', '') for elem in group if elem.get('text'))
            
            # Calculate encompassing bounding box
            min_x = min(elem['bbox'][0] for elem in group)
            min_y = min(elem['bbox'][1] for elem in group)
            max_x = max(elem['bbox'][2] for elem in group)
            max_y = max(elem['bbox'][3] for elem in group)
            
            merged_bbox = [min_x, min_y, max_x, max_y]
            merged_center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
            
            # Create merged element
            merged_elem = {
                'id': f'merged_text_{group_idx}',
                'page_num': group[0]['page_num'],
                'type': 'TEXT',
                'text': merged_text,
                'bbox': merged_bbox,
                'center': merged_center,
                'font_size': group[0].get('font_size'),
                'font_weight': group[0].get('font_weight')
            }
            
            merged_elements.append(merged_elem)
    
    return merged_elements + non_text_elements


def merge_input_fields(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge horizontally adjacent INPUT_FIELD and CHECKBOX elements.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        List with merged input field elements
    """
    input_types = ['INPUT_FIELD', 'CHECKBOX']
    input_elements = [elem for elem in elements if elem.get('type') in input_types]
    other_elements = [elem for elem in elements if elem.get('type') not in input_types]
    
    merged_groups = []
    used_indices = set()
    
    for i, elem in enumerate(input_elements):
        if i in used_indices:
            continue
            
        # Start a new merge group
        merge_group = [elem]
        used_indices.add(i)
        
        # Find all elements that should be merged with this one
        changed = True
        while changed:
            changed = False
            for j, other_elem in enumerate(input_elements):
                if j in used_indices:
                    continue
                    
                # Check if this element is adjacent to any element in current group
                for group_elem in merge_group:
                    if is_horizontally_adjacent(group_elem, other_elem, max_gap=5.0):
                        merge_group.append(other_elem)
                        used_indices.add(j)
                        changed = True
                        break
        
        merged_groups.append(merge_group)
    
    # Create merged elements
    merged_elements = []
    for group_idx, group in enumerate(merged_groups):
        if len(group) == 1:
            # Single element, keep as is
            merged_elements.append(group[0])
        else:
            # Merge multiple elements
            # Sort by x-coordinate
            group.sort(key=lambda x: x['bbox'][0])
            
            # Calculate encompassing bounding box
            min_x = min(elem['bbox'][0] for elem in group)
            min_y = min(elem['bbox'][1] for elem in group)
            max_x = max(elem['bbox'][2] for elem in group)
            max_y = max(elem['bbox'][3] for elem in group)
            
            merged_bbox = [min_x, min_y, max_x, max_y]
            merged_center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
            
            # Create merged element
            merged_elem = {
                'id': f'merged_field_{group_idx}',
                'page_num': group[0]['page_num'],
                'type': 'INPUT_FIELD',
                'text': None,  # Keep as null for merged input fields
                'bbox': merged_bbox,
                'center': merged_center
            }
            
            merged_elements.append(merged_elem)
    
    return merged_elements + other_elements


def calculate_distance(center1: List[float], center2: List[float]) -> float:
    """
    Calculate Euclidean distance between two center points.
    
    Args:
        center1, center2: Center coordinates [x, y]
        
    Returns:
        Euclidean distance rounded to 2 decimal places
    """
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    return round(math.sqrt(dx * dx + dy * dy), 2)


def calculate_direction(center1: List[float], center2: List[float]) -> str:
    """
    Calculate relative direction from center1 to center2.
    
    Args:
        center1, center2: Center coordinates [x, y]
        
    Returns:
        Direction string (e.g., 'right', 'top-left', 'below')
    """
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    
    # Determine primary directions
    horizontal = ''
    vertical = ''
    
    if abs(dx) > 5:  # Threshold for considering horizontal movement
        horizontal = 'right' if dx > 0 else 'left'
    
    if abs(dy) > 5:  # Threshold for considering vertical movement
        vertical = 'below' if dy > 0 else 'above'
    
    # Combine directions
    if horizontal and vertical:
        return f"{vertical}-{horizontal}"
    elif horizontal:
        return horizontal
    elif vertical:
        return vertical
    else:
        return 'same-position'


def calculate_alignment(elem1: Dict[str, Any], elem2: Dict[str, Any]) -> str:
    """
    Determine if two elements are horizontally or vertically aligned.
    
    Args:
        elem1, elem2: Element dictionaries with center coordinates
        
    Returns:
        Alignment string: 'horizontal', 'vertical', or 'none'
    """
    center1, center2 = elem1['center'], elem2['center']
    
    # Check horizontal alignment (similar Y coordinates)
    y_diff = abs(center1[1] - center2[1])
    if y_diff <= 5:  # Threshold for horizontal alignment
        return 'horizontal'
    
    # Check vertical alignment (similar X coordinates)
    x_diff = abs(center1[0] - center2[0])
    if x_diff <= 5:  # Threshold for vertical alignment
        return 'vertical'
    
    return 'none'


def find_nearest_neighbors(element: Dict[str, Any], all_elements: List[Dict[str, Any]], 
                          max_neighbors: int = 5) -> List[Dict[str, Any]]:
    """
    Find the nearest neighbors for a given element on the same page.
    
    Args:
        element: The target element
        all_elements: List of all elements to search through
        max_neighbors: Maximum number of neighbors to return
        
    Returns:
        List of neighbor dictionaries with relationship descriptors
    """
    same_page_elements = [
        elem for elem in all_elements 
        if elem.get('page_num') == element.get('page_num') and elem['id'] != element['id']
    ]
    
    # Calculate distances and create neighbor objects
    neighbors = []
    for other_elem in same_page_elements:
        distance = calculate_distance(element['center'], other_elem['center'])
        direction = calculate_direction(element['center'], other_elem['center'])
        alignment = calculate_alignment(element, other_elem)
        
        neighbor = {
            'id': other_elem['id'],
            'distance': distance,
            'direction': direction,
            'alignment': alignment
        }
        neighbors.append(neighbor)
    
    # Sort by distance and return top N
    neighbors.sort(key=lambda x: x['distance'])
    return neighbors[:max_neighbors]


def calculate_relationships(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate spatial relationships (Gestalt descriptors) for all elements.
    
    Args:
        elements: List of merged elements
        
    Returns:
        List of elements enriched with neighbor relationships
    """
    enriched_elements = []
    
    for element in elements:
        # Find nearest neighbors
        neighbors = find_nearest_neighbors(element, elements)
        
        # Add neighbors to element
        enriched_element = element.copy()
        enriched_element['neighbors'] = neighbors
        
        enriched_elements.append(enriched_element)
    
    return enriched_elements


def save_processed_elements(elements: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save processed elements to output JSON file.
    
    Args:
        elements: List of processed elements with relationships
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(elements, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"Could not write to output file {output_file}: {e}")


def process_elements(input_file: str, output_file: str) -> None:
    """
    Main processing function that orchestrates the entire workflow.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Load raw elements
    elements = load_elements(input_file)
    print(f"Loaded {len(elements)} raw elements from {input_file}")
    
    # Merge text fragments
    elements = merge_text_fragments(elements)
    print(f"After text merging: {len(elements)} elements")
    
    # Merge input fields
    elements = merge_input_fields(elements)
    print(f"After input field merging: {len(elements)} elements")
    
    # Calculate spatial relationships
    elements = calculate_relationships(elements)
    print(f"Added spatial relationships to {len(elements)} elements")
    
    # Save processed elements
    save_processed_elements(elements, output_file)
    print(f"Saved processed elements to {output_file}")


def main():
    """
    Command-line interface for the Gestalt pre-processor.
    """
    parser = argparse.ArgumentParser(
        description='IRIS Phase 1.5: Gestalt Pre-processor - Merge fragmented elements and add spatial relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gestalt_processor.py input.json output.json
  python gestalt_processor.py raw_elements.json processed_elements.json
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input JSON file containing raw elements from Phase 1'
    )
    
    parser.add_argument(
        'output_file', 
        help='Path to output JSON file for processed elements with spatial relationships'
    )
    
    parser.add_argument(
        '--max-neighbors',
        type=int,
        default=5,
        help='Maximum number of neighbors to calculate for each element (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        process_elements(args.input_file, args.output_file)
        print("✓ Gestalt pre-processing completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())