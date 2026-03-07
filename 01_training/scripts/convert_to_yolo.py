import os
import cv2
import numpy as np
import yaml
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

def create_yolo_dirs(output_dir, category):
    """Create YOLO dataset directory structure."""
    category_dir = Path(output_dir) / category
    
    dirs_to_create = [
        category_dir / 'images' / 'train',
        category_dir / 'images' / 'val',
        category_dir / 'labels' / 'train',
        category_dir / 'labels' / 'val'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return category_dir

def mask_to_bbox(mask):
    """Convert binary mask to bounding box in YOLO format."""
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (main defect area)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Normalize to image dimensions (assuming mask dimensions are image dimensions)
    img_height, img_width = mask.shape
    
    # Convert to YOLO format: center_x, center_y, width, height (all normalized 0-1)
    center_x = (x + w/2) / img_width
    center_y = (y + h/2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    
    return center_x, center_y, norm_width, norm_height

def resize_image(image, target_size):
    """Resize image while maintaining aspect ratio."""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    padded = np.zeros((target_size, target_size, 3) if len(image.shape) == 3 else (target_size, target_size), dtype=image.dtype)
    
    # Calculate padding offsets
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    # Place resized image in padded image
    if len(image.shape) == 3:
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    else:
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return padded, scale, x_offset, y_offset

def adjust_bbox_for_padding(bbox, scale, x_offset, y_offset, target_size):
    """Adjust bounding box coordinates after image resizing and padding."""
    center_x, center_y, width, height = bbox
    
    # Convert back to pixel coordinates (original image scale)
    orig_center_x = center_x * target_size / scale - x_offset / scale
    orig_center_y = center_y * target_size / scale - y_offset / scale
    orig_width = width * target_size / scale
    orig_height = height * target_size / scale
    
    # Normalize to new image size
    new_center_x = orig_center_x / target_size
    new_center_y = orig_center_y / target_size
    new_width = orig_width / target_size
    new_height = orig_height / target_size
    
    return new_center_x, new_center_y, new_width, new_height

def process_category(data_dir, category, output_dir, img_size):
    """Process a single category and convert to YOLO format."""
    print(f"Processing category: {category}")
    
    data_path = Path(data_dir)
    category_path = data_path / category
    category_output_dir = create_yolo_dirs(output_dir, category)
    
    # Check if category exists
    if not category_path.exists():
        print(f"Error: Category '{category}' not found in {data_path}")
        return None
    
    # Get defect types from ground_truth directory
    ground_truth_dir = category_path / 'ground_truth'
    defect_types = []
    
    if ground_truth_dir.exists():
        defect_types = [d.name for d in ground_truth_dir.iterdir() if d.is_dir()]
        print(f"Found defect types: {defect_types}")
    else:
        print(f"Warning: No ground_truth directory found for {category}")
    
    # Create class mapping
    class_mapping = {defect_type: idx for idx, defect_type in enumerate(defect_types)}
    
    stats = {
        'train_images': 0,
        'val_images': 0,
        'total_defects': 0
    }
    
    # Process training images (good images)
    train_dir = category_path / 'train' / 'good'
    if train_dir.exists():
        print(f"Processing training images from {train_dir}")
        for img_file in train_dir.glob('*.png'):
            # Load and resize image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            resized_img, _, _, _ = resize_image(image, img_size)
            
            # Save image
            output_img_path = category_output_dir / 'images' / 'train' / f"{img_file.stem}.jpg"
            cv2.imwrite(str(output_img_path), resized_img)
            
            # Create empty label file
            label_path = category_output_dir / 'labels' / 'train' / f"{img_file.stem}.txt"
            label_path.touch()
            
            stats['train_images'] += 1
    
    # Process test images with defects
    test_dir = category_path / 'test'
    if test_dir.exists():
        for defect_type in defect_types:
            test_defect_dir = test_dir / defect_type
            ground_truth_defect_dir = ground_truth_dir / defect_type
            
            if test_defect_dir.exists() and ground_truth_defect_dir.exists():
                print(f"Processing test images for defect type: {defect_type}")
                
                for img_file in test_defect_dir.glob('*.png'):
                    # Load test image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    # Find corresponding ground truth mask (with _mask suffix)
                    mask_filename = f"{img_file.stem}_mask.png"
                    mask_file = ground_truth_defect_dir / mask_filename
                    if not mask_file.exists():
                        print(f"Warning: No ground truth mask found for {mask_filename}")
                        continue
                    
                    # Load mask
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    
                    # Convert mask to bbox
                    bbox = mask_to_bbox(mask)
                    if bbox is None:
                        print(f"Warning: No contours found in mask {mask_file.name}")
                        continue
                    
                    # Resize image and adjust bbox
                    resized_img, scale, x_offset, y_offset = resize_image(image, img_size)
                    
                    # Save image
                    output_img_path = category_output_dir / 'images' / 'val' / f"{defect_type}_{img_file.stem}.jpg"
                    cv2.imwrite(str(output_img_path), resized_img)
                    
                    # Create label file
                    label_path = category_output_dir / 'labels' / 'val' / f"{defect_type}_{img_file.stem}.txt"
                    class_id = class_mapping[defect_type]
                    
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                    stats['val_images'] += 1
                    stats['total_defects'] += 1
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(category_output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {idx: name for name, idx in class_mapping.items()},
        'nc': len(defect_types)
    }
    
    yaml_path = category_output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\nConversion completed for {category}:")
    print(f"  - Training images: {stats['train_images']}")
    print(f"  - Validation images: {stats['val_images']}")
    print(f"  - Total defects detected: {stats['total_defects']}")
    print(f"  - Classes: {len(defect_types)} ({', '.join(defect_types)})")
    print(f"  - Output directory: {category_output_dir}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Convert MVTec dataset to YOLO format')
    parser.add_argument('--category', required=True, help='Category to convert (e.g. "bottle")')
    parser.add_argument('--output_dir', default='data/processed', help='Output directory for YOLO dataset')
    parser.add_argument('--img_size', type=int, default=640, help='Target image size for YOLO (default: 640)')
    parser.add_argument('--data_dir', default='data/raw', help='Path to MVTec raw data directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MVTec to YOLO Converter")
    print("=" * 60)
    print(f"Category: {args.category}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.img_size}")
    print("=" * 60)
    
    # Process the category
    stats = process_category(args.data_dir, args.category, args.output_dir, args.img_size)
    
    if stats:
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("CONVERSION FAILED")
        print("=" * 60)

if __name__ == "__main__":
    main()