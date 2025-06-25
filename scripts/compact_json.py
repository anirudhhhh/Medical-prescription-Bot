#!/usr/bin/env python3
"""
Script to convert all JSON outputs in datasets to compact format
Removes all whitespace and formatting for better training efficiency
"""

import json
import os
import glob
from pathlib import Path

def compact_json_outputs(dataset_path: str):
    """Convert JSON outputs in a dataset to compact format"""
    print(f"Processing {dataset_path}...")
    
    # Read all samples
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # Convert JSON outputs to compact format
    compacted_count = 0
    for sample in samples:
        try:
            # Parse the JSON output
            output_data = json.loads(sample['output'])
            # Re-serialize as compact JSON
            sample['output'] = json.dumps(output_data, separators=(',', ':'))
            compacted_count += 1
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON found in sample: {e}")
            continue
    
    # Write back to file
    with open(dataset_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"  ✅ Compacted {compacted_count} JSON outputs")
    return compacted_count

def main():
    """Convert all dataset files to compact JSON format"""
    print("🔧 Converting JSON outputs to compact format...")
    
    # Find all dataset files
    data_dir = Path("data")
    dataset_files = list(data_dir.glob("*.jsonl"))
    
    if not dataset_files:
        print("❌ No dataset files found in data/ directory")
        return
    
    total_compacted = 0
    for dataset_file in dataset_files:
        compacted = compact_json_outputs(str(dataset_file))
        total_compacted += compacted
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Total JSON outputs compacted: {total_compacted}")
    print(f"📁 Files processed: {len(dataset_files)}")
    
    # Show example of compacted format
    if dataset_files:
        print(f"\n📋 Example of compacted format:")
        with open(dataset_files[0], 'r') as f:
            sample = json.loads(f.readline().strip())
            print(f"Output length: {len(sample['output'])} characters")
            print(f"Sample: {sample['output'][:100]}...")

if __name__ == "__main__":
    main() 