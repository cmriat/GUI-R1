#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to show multiple samples from each parquet file in the GUI-R1 dataset.
"""

from datasets import load_dataset, Features, Value, Sequence
import os
import json
import pyarrow.parquet as pq
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Show multiple samples from parquet datasets')
    parser.add_argument('--data_dir', type=str, default="/root/codes/GUI-R1/datasets/GUI-R1",
                        help='Directory containing the parquet files')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to show from each dataset (default: 3)')
    parser.add_argument('--random', action='store_true',
                        help='Select random samples instead of sequential ones')
    parser.add_argument('--save_json', action='store_true',
                        help='Save samples to a JSON file')
    parser.add_argument('--json_path', type=str, default="multiple_samples.json",
                        help='Path to save the JSON file (default: multiple_samples.json)')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Specific dataset to show samples from (without .parquet extension) or "all"')
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

def format_sample(sample_dict, index):
    """Format a sample for display"""
    output = [f"Sample #{index}:"]
    output.append("-" * 40)
    
    # Order keys for consistent display
    ordered_keys = []
    
    # First display instruction and history if available
    if 'instruction' in sample_dict:
        ordered_keys.append('instruction')
    if 'history' in sample_dict:
        ordered_keys.append('history')
    
    # Then display action-related fields
    if 'gt_action' in sample_dict:
        ordered_keys.append('gt_action')
    if 'gt_bbox' in sample_dict:
        ordered_keys.append('gt_bbox')
    if 'gt_input_text' in sample_dict:
        ordered_keys.append('gt_input_text')
    
    # Then display metadata
    for k in sample_dict.keys():
        if k not in ordered_keys and k != 'image':
            ordered_keys.append(k)
    
    # Add image at the end
    if 'image' in sample_dict:
        ordered_keys.append('image')
    
    # Format each field
    for k in ordered_keys:
        v = sample_dict[k]
        if k == 'instruction':
            output.append(f"Instruction: {v}")
        elif k == 'history':
            if v and v.strip() and v.strip() != 'None':
                output.append(f"History: {v}")
        elif k == 'gt_action':
            output.append(f"Action: {v}")
        elif k == 'gt_bbox':
            output.append(f"Bounding Box: {v}")
        elif k == 'gt_input_text':
            if v != 'no input text':
                output.append(f"Input Text: {v}")
        elif k == 'image':
            output.append("Image: <binary data>")
        else:
            output.append(f"{k}: {v}")
    
    return "\n".join(output)

def main():
    args = parse_arguments()
    
    # Define the base directory
    base_dir = args.data_dir
    
    # Get all parquet files in the directory
    parquet_files = [f for f in os.listdir(base_dir) if f.endswith('.parquet')]
    parquet_files.sort()  # Sort for consistent output
    
    # Filter to specific dataset if requested
    if args.dataset != 'all':
        dataset_file = f"{args.dataset}.parquet"
        if dataset_file in parquet_files:
            parquet_files = [dataset_file]
        else:
            print(f"Error: Dataset '{args.dataset}' not found. Available datasets:")
            for f in parquet_files:
                print(f"  - {os.path.splitext(f)[0]}")
            return
    
    print(f"Found {len(parquet_files)} parquet files in {base_dir}")
    print(f"Showing {args.num_samples} samples from each dataset")
    
    # Dictionary to store samples
    all_samples = {}
    
    # Process each parquet file
    for file_name in parquet_files:
        file_path = os.path.join(base_dir, file_name)
        dataset_name = os.path.splitext(file_name)[0]  # Remove .parquet extension
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        dataset = load_parquet_safely(file_path, dataset_name)
        
        if dataset is not None and len(dataset) > 0:
            # Get sample indices
            dataset_size = len(dataset)
            if args.random:
                # Random sampling
                indices = random.sample(range(dataset_size), min(args.num_samples, dataset_size))
            else:
                # Sequential sampling
                indices = list(range(min(args.num_samples, dataset_size)))
            
            # Store samples for this dataset
            all_samples[dataset_name] = []
            
            # Show each sample
            for i, idx in enumerate(indices):
                sample = dataset[idx]
                sample_dict = get_sample_dict(sample)
                
                # Format and print the sample
                formatted_sample = format_sample(sample_dict, idx)
                print(formatted_sample)
                print()
                
                # Store sample for JSON output
                all_samples[dataset_name].append(sample_dict)
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
