#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to show one sample from each parquet file in the GUI-R1 dataset.
"""

from datasets import load_dataset, Features, Value, Sequence
import os
import json
import pyarrow.parquet as pq
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Show samples from parquet datasets')
    parser.add_argument('--data_dir', type=str, default="/root/codes/GUI-R1/datasets/GUI-R1",
                        help='Directory containing the parquet files')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of the sample to show (default: 0)')
    parser.add_argument('--save_json', action='store_true',
                        help='Save samples to a JSON file')
    parser.add_argument('--json_path', type=str, default="samples.json",
                        help='Path to save the JSON file (default: samples.json)')
    return parser.parse_args()

def load_parquet_safely(file_path, file_name):
    """Load a parquet file with custom schema"""
    try:
        # Read the schema first to determine the features
        schema = pq.read_schema(file_path)
        
        # Create a features dictionary based on the schema
        features = {}
        for field in schema.names:
            if field == 'image':
                features[field] = {'bytes': Value(dtype='binary')}
            elif field == 'gt_bbox':
                # Handle both int64 and double types
                features[field] = Sequence(feature=Value(dtype='float64'), length=-1)
            else:
                features[field] = Value(dtype='string')
        
        # Load the dataset with the custom features
        dataset = load_dataset(
            "parquet",
            data_files={file_name: file_path},
            split=file_name,
            features=Features(features)
        )
        
        return dataset
    
    except Exception as e:
        print(f"Error loading {file_name} from {file_path}: {e}")
        return None

def get_sample_dict(sample):
    """Convert sample to a dictionary without binary image data"""
    sample_dict = {}
    for k, v in sample.items():
        if k == 'image':
            sample_dict[k] = "<binary data>"
        elif k == 'gt_bbox':
            sample_dict[k] = v
        else:
            sample_dict[k] = v
    return sample_dict

def main():
    args = parse_arguments()
    
    # Define the base directory
    base_dir = args.data_dir
    
    # Get all parquet files in the directory
    parquet_files = [f for f in os.listdir(base_dir) if f.endswith('.parquet')]
    parquet_files.sort()  # Sort for consistent output
    
    print(f"Found {len(parquet_files)} parquet files in {base_dir}")
    
    # Dictionary to store samples
    all_samples = {}
    
    # Process each parquet file
    for file_name in parquet_files:
        file_path = os.path.join(base_dir, file_name)
        dataset_name = os.path.splitext(file_name)[0]  # Remove .parquet extension
        
        print(f"\n{'='*80}")
        print(f"Sample from {dataset_name}:")
        print(f"{'='*80}")
        
        dataset = load_parquet_safely(file_path, dataset_name)
        
        if dataset is not None and len(dataset) > 0:
            # Get sample at the specified index or default to 0 if out of range
            sample_index = min(args.sample_index, len(dataset) - 1)
            sample = dataset[sample_index]
            
            # Convert to dictionary without binary image data
            sample_dict = get_sample_dict(sample)
            
            # Print sample in a readable format
            for k, v in sample_dict.items():
                if k == 'gt_bbox':
                    print(f"{k}: {v}")
                else:
                    print(f"{k}: {v}")
            
            # Store sample for JSON output
            all_samples[dataset_name] = sample_dict
        else:
            print(f"No samples available for {dataset_name}")
    
    # Save samples to JSON file if requested
    if args.save_json:
        try:
            with open(args.json_path, 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)
            print(f"\nSamples saved to {args.json_path}")
        except Exception as e:
            print(f"Error saving samples to JSON: {e}")

if __name__ == "__main__":
    main()
