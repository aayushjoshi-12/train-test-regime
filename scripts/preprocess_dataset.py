import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict


def download_and_process_dataset(
    url, output_dir, train_size=0.8, val_size=0.1, test_size=0.1
):
    """
    Download a dataset from a URL, split it into train, validation, and test sets,
    and format each using the format_data function.

    Args:
        url: URL of the CSV dataset
        output_dir: Directory to save the processed datasets
        train_size, val_size, test_size: Split proportions (should sum to 1)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download and load the CSV
    print(f"Downloading dataset from {url}")
    df = pd.read_csv(url)

    # Create a Dataset object
    dataset = Dataset.from_pandas(df)

    # Split the dataset
    splits = dataset.train_test_split(
        test_size=(val_size + test_size), shuffle=True, seed=42
    )

    # Further split the test set into validation and test
    test_val_split = splits["test"].train_test_split(
        test_size=test_size / (test_size + val_size), shuffle=True, seed=42
    )

    # Create the final dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "validation": test_val_split["train"],
            "test": test_val_split["test"],
        }
    )

    # Format each split using format_data
    for split_name, ds in dataset_dict.items():
        formatted_ds = ds.map(format_data)
        formatted_ds.save_to_disk(os.path.join(output_dir, split_name))
        print(f"Processed {split_name} set: {len(formatted_ds)} examples")

    return dataset_dict


def format_data(row):
    return {
        "text": f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{row["response"]}<|eot_id|><|end_of_text|>
"""
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess a dataset")
    parser.add_argument("--url", type=str, required=True, help="URL to the CSV dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--train_size", type=float, default=0.8, help="Proportion of data for training"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Proportion of data for validation"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Proportion of data for testing"
    )

    args = parser.parse_args()

    # Verify that proportions sum to 1
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Split proportions must sum to 1, got {total}")

    download_and_process_dataset(
        args.url, args.output_dir, args.train_size, args.val_size, args.test_size
    )
