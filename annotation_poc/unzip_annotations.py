#!/usr/bin/env python3
"""
Unzip annotation files into separate directories based on their names.
Each zip file is extracted into a directory with the same name (without .zip extension).
"""

import argparse
import os
import zipfile
from pathlib import Path


def unzip_annotations(folder_path: str, rm_zip: bool = False) -> None:
    """
    Unzip all .zip files in the specified folder into separate directories.

    Args:
        folder_path: Path to the folder containing zip files
        rm_zip: If True, remove zip files after extraction (default: False)
    """
    folder = Path(folder_path).expanduser().resolve()

    if not folder.exists():
        print(f"Error: Directory '{folder}' does not exist")
        return

    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory")
        return

    # Find all zip files
    zip_files = list(folder.glob("*.zip"))

    if not zip_files:
        print(f"No .zip files found in '{folder}'")
        return

    print(f"Found {len(zip_files)} zip file(s) in '{folder}'")

    for zip_file in zip_files:
        # Create directory name by removing .zip extension
        target_dir = folder / zip_file.stem

        print(f"Extracting '{zip_file.name}' to '{target_dir.name}/'...")

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract zip file
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  ✓ Extracted successfully")

            # Remove zip file if requested
            if rm_zip:
                zip_file.unlink()
                print(f"  ✓ Removed '{zip_file.name}'")

        except zipfile.BadZipFile:
            print(f"  ✗ Error: '{zip_file.name}' is not a valid zip file")
        except Exception as e:
            print(f"  ✗ Error extracting '{zip_file.name}': {e}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Unzip annotation files into separate directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/Downloads/b12_annotations
  %(prog)s ~/Downloads/b12_annotations --rm-zip
  %(prog)s /path/to/annotations --rm-zip
        """
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to the folder containing zip files"
    )

    parser.add_argument(
        "--rm-zip",
        action="store_true",
        default=False,
        help="Remove zip archives after extraction (default: False)"
    )

    args = parser.parse_args()

    unzip_annotations(args.path, args.rm_zip)


if __name__ == "__main__":
    main()
