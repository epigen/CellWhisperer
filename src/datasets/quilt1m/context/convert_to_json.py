#!/usr/bin/env python3
"""
Convert quilt1m_examples.txt (CSV) to JSON format.
"""

import csv
import json
import ast


def safe_eval(value):
    """Safely evaluate string representations of Python literals."""
    if not value or value.strip() == "":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If it can't be parsed as a literal, return as string
        return value


def convert_csv_to_json(input_file, output_file):
    """Convert CSV file to JSON format."""
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Convert specific fields that should be parsed as Python literals
            if 'pathology' in row:
                row['pathology'] = safe_eval(row['pathology'])
            if 'roi_text' in row:
                row['roi_text'] = safe_eval(row['roi_text'])
            if 'med_umls_ids' in row:
                row['med_umls_ids'] = safe_eval(row['med_umls_ids'])
            
            # Convert numeric fields
            for field in ['magnification', 'height', 'width', 'not_histology', 'single_wsi']:
                if field in row and row[field]:
                    try:
                        if field in ['not_histology', 'single_wsi']:
                            row[field] = int(float(row[field]))  # Handle cases like "0" or "0.0"
                        else:
                            row[field] = float(row[field])
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
            
            # Remove the 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in row:
                del row['Unnamed: 0']
            
            data.append(row)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} records from {input_file} to {output_file}")


if __name__ == "__main__":
    input_file = "quilt1m_examples.txt"
    output_file = "quilt1m_examples.json"
    convert_csv_to_json(input_file, output_file)