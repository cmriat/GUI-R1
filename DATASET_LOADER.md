# GUI-R1 Dataset Loader

This document explains how to use the `test.py` script for loading and analyzing the GUI-R1 dataset, which consists of multiple parquet files with different schemas.

## Problem

The original code was failing with the error:

```
datasets.table.CastError: Couldn't cast
image: struct<bytes: binary>
  child 0, bytes: binary
history: string
instruction: string
gt_action: string
gt_bbox: list<element: int64>
  child 0, element: int64
gt_input_text: string
group: string
ui_type: string
to
{'image': {'bytes': Value(dtype='binary', id=None)}, 'gt_bbox': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), 'instruction': Value(dtype='string', id=None), 'id': Value(dtype='string', id=None), 'gt_action': Value(dtype='string', id=None), 'gt_input_text': Value(dtype='string', id=None), 'history': Value(dtype='string', id=None), 'task_type': Value(dtype='string', id=None)}
because column names don't match
```

This error occurs because the different parquet files in the dataset have different schemas:
- Some files have `id` and `task_type` columns
- Others have `group` and `ui_type` columns
- The `gt_bbox` column has different data types (double vs int64)

## Solution

The `test.py` script provides a robust solution for loading these parquet files by:
1. Reading the schema of each file first
2. Creating a custom features dictionary based on the schema
3. Loading the dataset with the custom features
4. Handling type conversions (e.g., int64 to float64 for bbox coordinates)

## Usage

```bash
# Load all datasets
python test.py

# Load a specific dataset
python test.py --dataset train

# Load a specific dataset with verbose output
python test.py --dataset androidcontrol_high_test --verbose

# Specify a different data directory
python test.py --data_dir /path/to/datasets
```

### Command-line Arguments

- `--dataset`: Specific dataset to load (without .parquet extension) or "all" to load all datasets
- `--verbose`: Show detailed information including schema and sample data
- `--data_dir`: Directory containing the parquet files

## Dataset Information

The GUI-R1 dataset contains the following parquet files:
- train.parquet
- test.parquet
- androidcontrol_high_test.parquet
- androidcontrol_low_test.parquet
- guiact_web_test.parquet
- guiodyssey_test.parquet
- omniact_desktop_test.parquet
- omniact_web_test.parquet
- screenspot_pro_test.parquet
- screenspot_test.parquet

Each dataset has slightly different schemas, but they all contain:
- `image`: Binary data of the screenshot
- `instruction`: Text instruction for the task
- `gt_bbox`: Bounding box coordinates for the target UI element

## Example Output

When running with the verbose option, you'll see detailed information about each dataset:

```
Schema for train:
image: struct<bytes: binary>
  child 0, bytes: binary
gt_bbox: list<element: double>
  child 0, element: double
instruction: string
id: string
gt_action: string
gt_input_text: string
history: string
task_type: string

Successfully loaded train dataset with 3570 examples
Sample from train:
{'gt_bbox': [0.752, 0.889, 0.817, 0.932], 'instruction': 'click the UI element plateforme', 'id': '106068', 'gt_action': 'click', 'gt_input_text': 'no input text', 'history': 'None', 'task_type': 'low'}
Available columns: ['image', 'gt_bbox', 'instruction', 'id', 'gt_action', 'gt_input_text', 'history', 'task_type']

Analysis of train dataset:
  - Number of examples: 3570
  - Columns: ['image', 'gt_bbox', 'instruction', 'id', 'gt_action', 'gt_input_text', 'history', 'task_type']
  - Instruction length: avg=54.7, min=2, max=218
```

## Integration with Training Code

To use this dataset loader in your training code, you can import the `load_parquet_safely` function:

```python
from test import load_parquet_safely

# Load the training dataset
train_path = "/path/to/datasets/train.parquet"
train_data = load_parquet_safely(train_path, "train")

# Use the dataset for training
if train_data is not None:
    # Your training code here
    pass
```
