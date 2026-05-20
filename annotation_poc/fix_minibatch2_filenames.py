#!/usr/bin/env python3
"""
Fix malformed filenames in minibatch2 and update the XML annotations.

Fixes two naming issues:
1. Double underscore: 4.2_H_cam__269_menivimdesks_108.tif → 4.2_H_cam_269_menivimdesks_108.tif
2. Wrong field order: 4.2_H_cam_menivimdesk_226_11.tif → 4.2_H_cam_226_menivimdesk_11.tif

Expected format: 4.2_H_cam_<id>_<scene>_<frame>.tif
"""

import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
ROOT_PTH = "/home/klymy/data/cam4_revB"
BATCH_DIR = os.path.join(ROOT_PTH, "4.2_minibatch2")
CORRECTED_DIR = os.path.join(ROOT_PTH, "minibatch2_corrected")
XML_PATH = os.path.join(CORRECTED_DIR, "minibatch2_corrected.xml")
BACKUP_XML_PATH = os.path.join(CORRECTED_DIR, "minibatch2_corrected.xml.backup")

# Patterns to fix
DOUBLE_UNDERSCORE_PATTERN = re.compile(r'^4\.2_H_cam__(\d+)_(.+)_(\d+)\.tif$')
WRONG_ORDER_PATTERN = re.compile(r'^4\.2_H_cam_([a-z]+)_(\d+)_(\d+)\.tif$')

def analyze_filename(filename: str) -> Tuple[str, str, str]:
    """
    Analyze a filename and return (status, original, corrected).

    Returns:
        Tuple of (status, original_name, corrected_name)
        status: 'double_underscore', 'wrong_order', 'valid', or 'unknown'
    """
    # Check for double underscore pattern: 4.2_H_cam__269_menivimdesks_108.tif
    match = DOUBLE_UNDERSCORE_PATTERN.match(filename)
    if match:
        cam_id, scene, frame = match.groups()
        corrected = f"4.2_H_cam_{cam_id}_{scene}_{frame}.tif"
        return ('double_underscore', filename, corrected)

    # Check for wrong order pattern: 4.2_H_cam_menivimdesk_226_11.tif
    match = WRONG_ORDER_PATTERN.match(filename)
    if match:
        scene, cam_id, frame = match.groups()
        corrected = f"4.2_H_cam_{cam_id}_{scene}_{frame}.tif"
        return ('wrong_order', filename, corrected)

    # Check if it matches the expected pattern
    if re.match(r'^4\.2_H_cam_\d+_.+_\d+\.tif$', filename):
        return ('valid', filename, filename)

    return ('unknown', filename, filename)

def find_malformed_files(batch_dir: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Find all malformed files in the batch directory.

    Returns:
        Dict mapping status to list of (status, original, corrected) tuples
    """
    results = {
        'double_underscore': [],
        'wrong_order': [],
        'valid': [],
        'unknown': []
    }

    for filename in sorted(os.listdir(batch_dir)):
        if not filename.endswith('.tif'):
            continue

        status, original, corrected = analyze_filename(filename)
        results[status].append((status, original, corrected))

    return results

def rename_files(batch_dir: str, files_to_rename: List[Tuple[str, str, str]], dry_run: bool = False) -> List[Tuple[str, str]]:
    """
    Rename files in the batch directory.

    Returns:
        List of (old_name, new_name) tuples that were renamed
    """
    renamed = []

    for status, old_name, new_name in files_to_rename:
        if old_name == new_name:
            continue

        old_path = os.path.join(batch_dir, old_name)
        new_path = os.path.join(batch_dir, new_name)

        if not os.path.exists(old_path):
            print(f"  ⚠ Warning: File not found: {old_name}")
            continue

        if os.path.exists(new_path):
            print(f"  ⚠ Warning: Target already exists: {new_name}")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would rename: {old_name} → {new_name}")
        else:
            os.rename(old_path, new_path)
            print(f"  ✓ Renamed: {old_name} → {new_name}")

        renamed.append((old_name, new_name))

    return renamed

def update_xml_references(xml_path: str, rename_map: Dict[str, str], dry_run: bool = False) -> int:
    """
    Update image name references in the XML file.

    Args:
        xml_path: Path to the XML file
        rename_map: Dict mapping old filename to new filename
        dry_run: If True, don't write changes

    Returns:
        Number of references updated
    """
    # Create backup
    if not dry_run and os.path.exists(xml_path):
        shutil.copy2(xml_path, BACKUP_XML_PATH)
        print(f"  ✓ Backup created: {BACKUP_XML_PATH}")

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    updated_count = 0

    # Update image elements
    for image in root.findall('.//image'):
        original_name = image.get('name', '')

        # Extract just the filename from the path (e.g., "4.2_minibatch2/file.tif" → "file.tif")
        if '/' in original_name:
            prefix, filename = original_name.rsplit('/', 1)
        else:
            prefix = ""
            filename = original_name

        # Check if this filename needs to be updated
        if filename in rename_map:
            new_filename = rename_map[filename]
            new_name = f"{prefix}/{new_filename}" if prefix else new_filename

            if dry_run:
                print(f"  [DRY RUN] Would update XML: {original_name} → {new_name}")
            else:
                image.set('name', new_name)
                print(f"  ✓ Updated XML: {original_name} → {new_name}")

            updated_count += 1

    # Write updated XML
    if not dry_run and updated_count > 0:
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        print(f"  ✓ XML file updated: {xml_path}")

    return updated_count

def main():
    print("=" * 80)
    print("MiniBatch2 Filename Correction Script")
    print("=" * 80)
    print()

    # Check batch directory exists
    if not os.path.exists(BATCH_DIR):
        print(f"ERROR: Batch directory not found: {BATCH_DIR}")
        return 1

    # Check XML exists
    if not os.path.exists(XML_PATH):
        print(f"ERROR: XML file not found: {XML_PATH}")
        print(f"       Run fix_minibatch2_annotations.py first!")
        return 1

    print(f"Batch directory: {BATCH_DIR}")
    print(f"XML file:        {XML_PATH}")
    print()

    # Analyze files
    print("Analyzing files...")
    results = find_malformed_files(BATCH_DIR)

    total_files = sum(len(files) for files in results.values())
    malformed_files = results['double_underscore'] + results['wrong_order']

    print(f"  Total files:            {total_files}")
    print(f"  Valid files:            {len(results['valid'])}")
    print(f"  Double underscore:      {len(results['double_underscore'])}")
    print(f"  Wrong order:            {len(results['wrong_order'])}")
    print(f"  Unknown pattern:        {len(results['unknown'])}")
    print(f"  Files to fix:           {len(malformed_files)}")
    print()

    if results['unknown']:
        print("⚠ Warning: Unknown pattern files found:")
        for _, filename, _ in results['unknown'][:5]:
            print(f"  - {filename}")
        if len(results['unknown']) > 5:
            print(f"  ... and {len(results['unknown']) - 5} more")
        print()

    if not malformed_files:
        print("✓ No malformed files found. Nothing to do.")
        return 0

    # Show examples
    print("Examples of corrections:")
    examples = malformed_files[:5]
    for status, old_name, new_name in examples:
        print(f"  {status:18s}: {old_name}")
        print(f"  {'':18s}→ {new_name}")
        print()

    if len(malformed_files) > 5:
        print(f"  ... and {len(malformed_files) - 5} more files")
        print()

    # Confirm action
    print("=" * 80)
    response = input("Proceed with renaming files and updating XML? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0

    print()
    print("Renaming files...")
    renamed = rename_files(BATCH_DIR, malformed_files, dry_run=False)
    print(f"  ✓ Renamed {len(renamed)} files")
    print()

    # Build rename map for XML update
    rename_map = {old: new for old, new in renamed}

    print("Updating XML references...")
    updated_count = update_xml_references(XML_PATH, rename_map, dry_run=False)
    print(f"  ✓ Updated {updated_count} XML references")
    print()

    print("=" * 80)
    print("✓ Correction completed successfully!")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Renamed {len(renamed)} image files")
    print(f"  - Updated {updated_count} XML references")
    print(f"  - XML backup: {BACKUP_XML_PATH}")
    print()
    print("Next steps:")
    print("  1. Re-run the batch injection pipeline")
    print("  2. The anti-leakage validation should now pass")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
