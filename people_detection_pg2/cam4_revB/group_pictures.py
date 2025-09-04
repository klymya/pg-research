#!/usr/bin/env python3
"""
Script to group images by name pattern and create videos from each group.

Example filename pattern: 4.2_batch2____4.2_H_cam_272_spacesbooth_52.jpg
- Group: 4.2_batch2____4.2_H_cam_272_spacesbooth
- Index: 52

Images are sorted by index (as integers) and combined into AVI videos.
Optionally processes YOLO format annotations and saves them as JSON.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2


def parse_filename(filename):
    """
    Parse filename to extract group name and index.

    Args:
        filename: Image filename like '4.2_batch2____4.2_H_cam_272_spacesbooth_52.jpg'

    Returns:
        tuple: (group_name, index) or (None, None) if parsing fails
    """
    # Remove extension
    name_without_ext = Path(filename).stem

    # Pattern: everything up to the last underscore followed by digits
    match = re.match(r'^(.+)_(\d+)$', name_without_ext)
    if match:
        group_name = match.group(1)
        index = int(match.group(2))
        return group_name, index

    return None, None


def parse_yolo_annotation(annotation_line):
    """
    Parse a YOLO format annotation line.

    Args:
        annotation_line: Line like "0 0.7915546874999999 0.6777708333333333 0.22048437499999995 0.2962916666666666 0.71978125 0.5877083333333334"

    Returns:
        dict: {"box": [x, y, w, h], "keypoint": [x, y]} or None if parsing fails
    """
    parts = annotation_line.strip().split()
    if len(parts) < 6:
        return None

    try:
        # Skip class (always 0 for person)
        box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        keypoint = [float(parts[5]), float(parts[6])]
        return {"box": box, "keypoint": keypoint}
    except (ValueError, IndexError):
        return None


def get_image_groups(input_folder):
    """
    Group images by their name pattern.

    Args:
        input_folder: Path to folder containing images

    Returns:
        dict: {group_name: [(filename, index), ...]}
    """
    groups = defaultdict(list)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for filename in os.listdir(input_folder):
        file_path = Path(input_folder) / filename

        # Skip if not a file or not an image
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in image_extensions:
            continue

        group_name, index = parse_filename(filename)
        if group_name is not None and index is not None:
            groups[group_name].append((filename, index))
        else:
            print(f"Warning: Could not parse filename pattern for '{filename}'")

    # Sort each group by index
    for group_name in groups:
        groups[group_name].sort(key=lambda x: x[1])

    return groups


def get_annotation_groups(annotations_folder):
    """
    Group annotation files by their name pattern.

    Args:
        annotations_folder: Path to folder containing YOLO annotation files

    Returns:
        dict: {group_name: [(filename, index), ...]}
    """
    groups = defaultdict(list)

    for filename in os.listdir(annotations_folder):
        file_path = Path(annotations_folder) / filename

        # Skip if not a file or not a .txt file
        if not file_path.is_file() or file_path.suffix.lower() != '.txt':
            continue

        group_name, index = parse_filename(filename)
        if group_name is not None and index is not None:
            groups[group_name].append((filename, index))
        else:
            print(f"Warning: Could not parse annotation filename pattern for '{filename}'")

    # Sort each group by index
    for group_name in groups:
        groups[group_name].sort(key=lambda x: x[1])

    return groups


def load_annotations_for_group(annotation_list, annotations_folder):
    """
    Load annotations for a group of files.

    Args:
        annotation_list: List of (filename, index) tuples sorted by index
        annotations_folder: Path to annotations folder

    Returns:
        list: List of frame annotations in order
    """
    frame_annotations = []

    for filename, _ in annotation_list:
        annotation_path = Path(annotations_folder) / filename
        frame_objects = []

        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = parse_yolo_annotation(line)
                        if obj:
                            frame_objects.append(obj)

            frame_annotations.append({
                "frame_idx": len(frame_annotations),
                "objects": frame_objects
            })
        except FileNotFoundError:
            print(f"Warning: Annotation file not found: {annotation_path}")
            frame_annotations.append({
                "frame_idx": len(frame_annotations),
                "objects": []
            })
        except Exception as e:
            print(f"Warning: Error reading annotation file {annotation_path}: {e}")
            frame_annotations.append({
                "frame_idx": len(frame_annotations),
                "objects": []
            })

    return frame_annotations


def create_video_from_images(image_list, input_folder, output_path, fps=1):
    """
    Create AVI video from list of images.

    Args:
        image_list: List of (filename, index) tuples sorted by index
        input_folder: Path to input folder
        output_path: Path for output video file
        fps: Frames per second for the video
    """
    if not image_list:
        print(f"Warning: No images to create video for {output_path}")
        return False

    # Read first image to get dimensions
    first_image_path = Path(input_folder) / image_list[0][0]
    first_frame = cv2.imread(str(first_image_path))

    if first_frame is None:
        print(f"Error: Could not read first image {first_image_path}")
        return False

    height, width = first_frame.shape[:2]

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    # Write frames to video
    frames_written = 0
    for filename, _ in image_list:
        image_path = Path(input_folder) / filename
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        # Resize frame if dimensions don't match
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)
        frames_written += 1

    video_writer.release()

    print(f"Created video: {output_path} ({frames_written} frames)")
    return True


def save_annotations_json(annotations_data, output_path):
    """
    Save annotations data to JSON file.

    Args:
        annotations_data: List of frame annotations
        output_path: Path for output JSON file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(annotations_data, f, indent=2)
        print(f"Created annotations: {output_path}")
        return True
    except Exception as e:
        print(f"Error: Could not save annotations to {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Group images by name pattern and create videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python group_pictures.py /path/to/images /path/to/output --fps 2
  python group_pictures.py /path/to/images /path/to/output --fps 2 --annotations /path/to/annotations

Filename pattern:
  4.2_batch2____4.2_H_cam_272_spacesbooth_52.jpg
  Group: 4.2_batch2____4.2_H_cam_272_spacesbooth
  Index: 52 (sorted as integer)

YOLO annotation format:
  0 0.7915546874999999 0.6777708333333333 0.22048437499999995 0.2962916666666666 0.71978125 0.5877083333333334
  Class (0=person), box (x,y,w,h), keypoint (x,y)
        """
    )

    parser.add_argument('input_folder',
                       help='Path to folder containing images')
    parser.add_argument('output_folder',
                       help='Path to output folder for videos')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second for output videos (default: 1)')
    parser.add_argument('--annotations', type=str,
                       help='Path to folder containing YOLO format annotation files')

    args = parser.parse_args()

    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_path}")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        sys.exit(1)

    # Validate annotations folder if provided
    annotations_path = None
    if args.annotations:
        annotations_path = Path(args.annotations)
        if not annotations_path.exists():
            print(f"Error: Annotations folder does not exist: {annotations_path}")
            sys.exit(1)
        if not annotations_path.is_dir():
            print(f"Error: Annotations path is not a directory: {annotations_path}")
            sys.exit(1)

    # Create output folder if it doesn't exist
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing images from: {input_path}")
    print(f"Output videos to: {output_path}")
    print(f"FPS: {args.fps}")
    if annotations_path:
        print(f"Processing annotations from: {annotations_path}")
    print()

    # Group images
    image_groups = get_image_groups(input_path)

    if not image_groups:
        print("No image groups found!")
        sys.exit(1)

    # Group annotations if folder provided
    annotation_groups = {}
    if annotations_path:
        annotation_groups = get_annotation_groups(annotations_path)

    print(f"Found {len(image_groups)} image groups:")

    # Create videos and annotations for each group
    success_count = 0
    total_count = len(image_groups)

    for group_name, image_list in image_groups.items():
        print(f"  - {group_name}: {len(image_list)} images")

        # Create video filename
        video_filename = f"{group_name}.avi"
        video_path = output_path / video_filename

        # Create video
        video_success = create_video_from_images(image_list, input_path, video_path, args.fps)

        # Process annotations if available
        annotations_success = True
        if annotations_path and group_name in annotation_groups:
            annotation_list = annotation_groups[group_name]
            
            # Ensure annotation list matches image list length and order
            if len(annotation_list) != len(image_list):
                print(f"Warning: Annotation count ({len(annotation_list)}) doesn't match image count ({len(image_list)}) for group {group_name}")
            
            # Load annotations
            annotations_data = load_annotations_for_group(annotation_list, annotations_path)
            
            # Save annotations JSON
            json_filename = f"{group_name}.json"
            json_path = output_path / json_filename
            annotations_success = save_annotations_json(annotations_data, json_path)

        if video_success and annotations_success:
            success_count += 1

    print()
    print(f"Successfully created {success_count}/{total_count} video/annotation pairs")

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()