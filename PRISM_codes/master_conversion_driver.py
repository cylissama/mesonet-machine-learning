#!/usr/bin/env python3
# Driver code for processing multiple BIL files
# Cy Dixon 2025, KY MESONET @ WKU

'''
run with
python master_conversion_driver.py /Volumes/PRISMdata/PRISM_data/an/ppt/daily/2019
'''

import os
import argparse
import time
from datetime import datetime
from convert_bil_to_csv import main as process_bil_file


def process_directory(input_dir, output_dir, pattern=None, subsample=1, random_state=None):
    """
    Process all BIL files in a directory in alphanumeric order.

    Parameters:
    - input_dir (str): Directory containing BIL files to process
    - output_dir (str): Directory where CSV files will be saved
    - pattern (str, optional): Only process files matching this pattern (e.g., "2021")
    - subsample (int, optional): Subsampling factor for data reduction
    - random_state (int, optional): Random seed for reproducible processing

    Returns:
    - tuple: (Number of files processed, List of output files)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Track files processed and output files generated
    processed_count = 0
    output_files = []
    error_files = []

    # Record start time
    start_time = time.time()

    print(f"Starting to process BIL files in: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    print(f"Subsample factor: {subsample}")
    print("-" * 50)

    # Get all BIL files and sort them alphanumerically
    bil_files = [f for f in os.listdir(input_dir) if f.endswith(".bil")]

    # Natural sort for filenames (handles numbers properly)
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Sort files alphanumerically
    bil_files.sort(key=natural_sort_key)

    # Filter files by pattern if specified
    if pattern:
        bil_files = [f for f in bil_files if pattern in f]

    print(f"Found {len(bil_files)} BIL files to process")

    # Process files in order
    for file_index, filename in enumerate(bil_files):
        # Full path to the file
        bil_file_path = os.path.join(input_dir, filename)

        try:
            # Process the BIL file and prepare the data in one step
            print(f"Processing {file_index + 1}/{len(bil_files)}: {filename}")
            csv_output = process_bil_file(
                bil_file_path,
                output_dir,
                subsample,
                random_state=random_state
            )

            if csv_output:
                output_files.append(csv_output)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_files.append(filename)

    # Calculate processing time
    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {processed_count} files")
    print(f"Generated {len(output_files)} CSV files")
    if error_files:
        print(f"Failed to process {len(error_files)} files")
        for err_file in error_files:
            print(f" - {err_file}")
    print("=" * 50)

    return processed_count, output_files


def main():
    """
    Main function to parse command line arguments and process BIL files.
    """
    parser = argparse.ArgumentParser(description="Process multiple BIL files in a directory")
    parser.add_argument("input_dir", help="Directory containing BIL files to process")
    parser.add_argument("--output_dir", default=None,
                        help="Directory where CSV files will be saved (default: /Volumes/Mesonet/spring_ml/output_data/)")
    parser.add_argument("--pattern", default=None, help="Only process files matching this pattern (e.g., '2021')")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsampling factor to reduce data size (default: 1, no subsampling)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducible processing (default: 42)")

    args = parser.parse_args()

    # Use default output directory if not specified
    if args.output_dir is None:
        args.output_dir = "/Volumes/Mesonet/spring_ml/output_data/"

    # Print the runtime configuration
    print("\nBIL File Processing Script")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.pattern:
        print(f"Processing files matching: {args.pattern}")
    print(f"Random state: {args.random_state}")
    print("\n")

    # Process the directory
    process_directory(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.subsample,
        args.random_state
    )


if __name__ == "__main__":
    main()