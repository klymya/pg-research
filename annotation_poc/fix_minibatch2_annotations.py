#!/usr/bin/env python3
"""
Fix minibatch2 annotations.xml image paths to match the standard format.

Changes:
  FROM: minibatch2/4.2_H_cam_215_menivimCS_15.tif
  TO:   4.2_minibatch2/4.2_H_cam_215_menivimCS_15.tif

This matches the pattern used in other batches:
  - 4.2_batch10/4.2_H_cam_228_MenivimMR_1.tif
  - 4.2_batch12/4.2_H_cam_249_ViaSroom_100.tif
"""

import shutil
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Configuration
ROOT_PTH = "/home/klymy/data/cam4_revB"
SOURCE_XML = "/home/klymy/Downloads/minibatch2/annotations.xml"
SOURCE_JSON = "/home/klymy/Downloads/minibatch2/instances_default.json"

BATCH_NAME = "minibatch2"
CORRECTED_DIR = os.path.join(ROOT_PTH, f"{BATCH_NAME}_corrected")
OUTPUT_XML = os.path.join(CORRECTED_DIR, f"{BATCH_NAME}_corrected.xml")
OUTPUT_JSON = os.path.join(CORRECTED_DIR, f"{BATCH_NAME}_corrected.json")

def fix_xml_image_paths(input_xml: str, output_xml: str) -> dict:
    """
    Fix image name attributes in CVAT XML from 'minibatch2/...' to '4.2_minibatch2/...'

    Returns:
        dict: Statistics about the changes made
    """
    # Parse XML
    tree = ET.parse(input_xml)
    root = tree.getroot()

    # Track changes
    stats = {
        "total_images": 0,
        "fixed_images": 0,
        "unchanged_images": 0,
        "examples_before": [],
        "examples_after": []
    }

    # Find all image elements
    for image in root.findall('.//image'):
        stats["total_images"] += 1
        original_name = image.get('name', '')

        # Save first 3 examples
        if stats["total_images"] <= 3:
            stats["examples_before"].append(original_name)

        # Fix the path if it starts with "minibatch2/"
        if original_name.startswith('minibatch2/'):
            # Replace "minibatch2/" with "4.2_minibatch2/"
            new_name = original_name.replace('minibatch2/', '4.2_minibatch2/', 1)
            image.set('name', new_name)
            stats["fixed_images"] += 1

            # Save first 3 examples
            if len(stats["examples_after"]) < 3:
                stats["examples_after"].append(new_name)
        else:
            stats["unchanged_images"] += 1

    # Write corrected XML
    os.makedirs(os.path.dirname(output_xml), exist_ok=True)
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)

    return stats

def main():
    print("=" * 80)
    print("MiniBatch2 Annotations Correction Script")
    print("=" * 80)
    print()

    # Check source files exist
    if not os.path.exists(SOURCE_XML):
        print(f"ERROR: Source XML not found: {SOURCE_XML}")
        return 1

    print(f"Source XML:     {SOURCE_XML}")
    print(f"Source JSON:    {SOURCE_JSON}")
    print(f"Output dir:     {CORRECTED_DIR}")
    print(f"Output XML:     {OUTPUT_XML}")
    print(f"Output JSON:    {OUTPUT_JSON}")
    print()

    # Fix XML paths
    print("Fixing XML image paths...")
    stats = fix_xml_image_paths(SOURCE_XML, OUTPUT_XML)
    print(f"  ✓ Processed {stats['total_images']} images")
    print(f"  ✓ Fixed {stats['fixed_images']} image paths")
    print(f"  ✓ Unchanged {stats['unchanged_images']} images")
    print()

    if stats['examples_before']:
        print("Examples of changes:")
        for before, after in zip(stats['examples_before'], stats['examples_after']):
            print(f"  BEFORE: {before}")
            print(f"  AFTER:  {after}")
            print()

    # Copy JSON file (no changes needed)
    if os.path.exists(SOURCE_JSON):
        print("Copying JSON file...")
        shutil.copy2(SOURCE_JSON, OUTPUT_JSON)
        print(f"  ✓ Copied to {OUTPUT_JSON}")
    else:
        print(f"  ⚠ Warning: JSON file not found, skipping: {SOURCE_JSON}")

    print()
    print("=" * 80)
    print("✓ Correction completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Review corrected XML: {OUTPUT_XML}")
    print(f"  2. Use this XML in the batch injection pipeline")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
