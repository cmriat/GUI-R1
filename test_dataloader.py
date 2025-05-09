from datasets import load_dataset, Features, Value, Sequence
import os
import argparse
import pyarrow.parquet as pq

def parse_arguments():
    parser = argparse.ArgumentParser(description='Load and analyze parquet datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Specific dataset to load (without .parquet extension) or "all" to load all datasets')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information including schema and sample data')
    parser.add_argument('--data_dir', type=str, default="/root/codes/GUI-R1/datasets/GUI-R1",
                        help='Directory containing the parquet files')
    return parser.parse_args()

# Function to safely load a parquet file with custom schema
def load_parquet_safely(file_path, file_name, verbose=False):
    try:
        # Read the schema first to determine the features
        schema = pq.read_schema(file_path)
        if verbose:
            print(f"Schema for {file_name}:")
            print(schema)

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

        print(f"Successfully loaded {file_name} dataset with {len(dataset)} examples")
        if verbose and len(dataset) > 0:
            print(f"Sample from {file_name}:")
            sample = dataset[0]
            # Print sample without the binary image data to keep output clean
            sample_dict = {k: v for k, v in sample.items() if k != 'image'}
            print(sample_dict)
            print(f"Available columns: {dataset.column_names}")

        return dataset

    except Exception as e:
        print(f"Error loading {file_name} from {file_path}: {e}")
        return None

def analyze_dataset(dataset, name):
    """Perform basic analysis on the dataset"""
    if dataset is None:
        return

    print(f"\nAnalysis of {name} dataset:")
    print(f"  - Number of examples: {len(dataset)}")
    print(f"  - Columns: {dataset.column_names}")

    # Check for missing values in key columns
    for col in dataset.column_names:
        if col != 'image':  # Skip binary data
            missing = sum(1 for item in dataset if not item[col])
            if missing > 0:
                print(f"  - Missing values in '{col}': {missing} ({missing/len(dataset)*100:.2f}%)")

    # Check instruction length distribution if available
    if 'instruction' in dataset.column_names:
        lengths = [len(item['instruction']) for item in dataset]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"  - Instruction length: avg={avg_len:.1f}, min={min_len}, max={max_len}")

def main():
    args = parse_arguments()

    # Define the base directory
    base_dir = args.data_dir

    # Get all parquet files in the directory
    parquet_files = [f for f in os.listdir(base_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files in {base_dir}")

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

    # Load each parquet file
    datasets = {}
    for file_name in parquet_files:
        file_path = os.path.join(base_dir, file_name)
        dataset_name = os.path.splitext(file_name)[0]  # Remove .parquet extension

        if args.verbose:
            print(f"\n{'='*50}\nProcessing {file_name}...\n{'='*50}")
        else:
            print(f"\nProcessing {file_name}...")

        dataset = load_parquet_safely(file_path, dataset_name, args.verbose)
        datasets[dataset_name] = dataset

        # Analyze the dataset
        if dataset is not None:
            analyze_dataset(dataset, dataset_name)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF LOADED DATASETS")
    print("="*80)
    for name, dataset in datasets.items():
        if dataset is not None:
            print(f"{name}: {len(dataset)} examples, columns: {dataset.column_names}")
        else:
            print(f"{name}: Failed to load")

if __name__ == "__main__":
    main()
