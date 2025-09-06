
import json
import argparse
import os
import google.generativeai as genai

def call_llm_api(prompt):
    """
    Sends the master prompt to the Google Gemini API and returns the response.
    """
    # --- IMPORTANT ---
    # For this to work, you must have your Google API key set as an
    # environment variable named 'GOOGLE_API_KEY'.
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        # Using Gemini 1.5 Pro for its large context window and reasoning
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Configure the model to ensure the output is JSON
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        
        print("Sending prompt to Gemini 1.5 Pro... (This may take a moment)")
        response = model.generate_content(prompt, generation_config=generation_config)
        
        print("Received response from LLM.")
        return response.text
        
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return None

def main():
    """
    The main orchestrator for the Phase 2 Reasoning Agent.
    This script generates a JSON "fill plan" but does not write to the PDF.
    """
    parser = argparse.ArgumentParser(description="IRIS Phase 2: Generate a form-filling plan using an LLM.")
    parser.add_argument("--empty-form-structure", required=True, help="Path to the processed JSON of the empty form.")
    parser.add_argument("--golden-sample-structure", required=True, help="Path to the processed JSON of the golden sample.")
    parser.add_argument("--new-data", required=True, help="Path to the JSON file with new data to fill.")
    parser.add_argument("--output-plan", required=True, help="Path to save the output JSON fill plan.")
    args = parser.parse_args()

    # 1. Load the three JSON context files
    try:
        with open(args.empty_form_structure, 'r') as f:
            empty_form_structure = json.load(f)
        with open(args.golden_sample_structure, 'r') as f:
            golden_sample_structure = json.load(f)
        with open(args.new_data, 'r') as f:
            new_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required JSON file. {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse a JSON file. {e}")
        return

    # 2. Construct the Master Prompt
    master_prompt_template = f"""
    You are IRIS, an expert document processing agent. Your task is to analyze the structure of an empty form, learn the layout from a golden sample, and use new data to create a precise "fill plan." Your final output must be only a single, valid JSON array of objects and nothing else.

    **1. The Golden Sample (The "Answer Key"):**
    This JSON represents a perfectly filled-out form. Use it to understand the spatial relationship (direction, alignment) between a text label and its corresponding value.
    ```json
    {json.dumps(golden_sample_structure, indent=2)}
    ```

    **2. The Empty Form (The "Canvas"):**
    This JSON represents the new, empty form you need to fill. It contains all the labels and empty input fields with their locations and neighbors.
    ```json
    {json.dumps(empty_form_structure, indent=2)}
    ```

    **3. The New Data (The "Content"):**
    This JSON contains the new information that you must place onto the empty form.
    ```json
    {json.dumps(new_data, indent=2)}
    ```

    **Your Instructions:**
    1. For each key-value pair in the "Content" JSON, find the corresponding label in the "Answer Key" JSON to learn its layout pattern.
    2. Locate that same label in the "Canvas" JSON.
    3. Search that label's "neighbors" for an `INPUT_FIELD` or `CHECKBOX` that best matches the spatial pattern you learned from the "Answer Key".
    4. For checkboxes, if the content value is a specific choice (e.g., "Female", "Spouse", "No"), find the checkbox whose nearest text neighbor matches that choice. The text to write in the checkbox's bbox should be "✓".
    5. Create a JSON object for each piece of content containing `{{"text": "the_value_to_write", "bbox": [the_target_bbox]}}`.
    6. Return a single JSON array containing all these objects. Do not include any other text, explanations, or markdown formatting in your response.
    """

    # 3. Call the LLM API to get the reasoning result
    llm_response_str = call_llm_api(master_prompt_template)
    
    if not llm_response_str:
        print("Failed to get a response from the LLM. Exiting.")
        return
        
    # 4. Validate and save the LLM's response
    try:
        # We parse it here just to validate that the LLM returned good JSON
        fill_plan = json.loads(llm_response_str)
        
        # Save the validated, formatted JSON plan to the output file
        with open(args.output_plan, 'w') as f:
            json.dump(fill_plan, f, indent=2)
            
        print(f"Successfully generated and saved the fill plan to: {args.output_plan}")

    except json.JSONDecodeError:
        print("Error: The LLM response was not valid JSON. Cannot proceed.")
        print("--- LLM Response ---")
        print(llm_response_str)
        print("--------------------")
        # Optionally, save the invalid response for debugging
        with open(args.output_plan.replace('.json', '_error.txt'), 'w') as f:
            f.write(llm_response_str)
        return

if __name__ == "__main__":
    main()