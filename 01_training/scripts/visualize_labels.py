import os
import cv2
import numpy as np
import yaml
import argparse
import random
from pathlib import Path

def load_dataset_config(data_dir):
    """Load dataset configuration from dataset.yaml"""
    yaml_path = Path(data_dir) / 'dataset.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found in {data_dir}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_class_colors(num_classes):
    """Generate distinct colors for each class"""
    colors = []
    for i in range(num_classes):
        # Generate HSV colors with full saturation and value, varying hue
        hue = i * 180 // num_classes
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors

def yolo_to_pixel_coords(yolo_bbox, img_width, img_height):
    """Convert YOLO format bbox to pixel coordinates"""
    center_x, center_y, width, height = yolo_bbox
    
    # Convert to pixel coordinates
    x_center = center_x * img_width
    y_center = center_y * img_height
    box_width = width * img_width
    box_height = height * img_height
    
    # Calculate top-left corner
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)
    
    return x1, y1, x2, y2

def draw_bbox_with_label(image, bbox_coords, class_name, class_color, confidence=None):
    """Draw bounding box with class label on image"""
    x1, y1, x2, y2 = bbox_coords
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), class_color, 2)
    
    # Prepare label text
    if confidence is not None:
        label = f"{class_name}: {confidence:.2f}"
    else:
        label = class_name
    
    # Get text size for label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw label background
    cv2.rectangle(image, 
                  (x1, y1 - text_height - 10), 
                  (x1 + text_width + 10, y1), 
                  class_color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1 + 5, y1 - 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    return image

def load_and_visualize_sample(data_dir, image_filename, split, class_names, class_colors):
    """Load and visualize a single sample with its annotations"""
    data_path = Path(data_dir)
    
    # Load image
    img_path = data_path / 'images' / split / image_filename
    if not img_path.exists():
        print(f"Warning: Image {img_path} not found")
        return None, 0
    
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Warning: Could not load image {img_path}")
        return None, 0
    
    # Load label file
    label_filename = img_path.stem + '.txt'
    label_path = data_path / 'labels' / split / label_filename
    
    num_boxes = 0
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        img_height, img_width = image.shape[:2]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                bbox_coords = yolo_to_pixel_coords([center_x, center_y, width, height], 
                                                   img_width, img_height)
                
                # Get class name and color
                class_name = class_names.get(class_id, f"class_{class_id}")
                class_color = class_colors[class_id % len(class_colors)]
                
                # Draw bounding box and label
                image = draw_bbox_with_label(image, bbox_coords, class_name, class_color)
                num_boxes += 1
    
    return image, num_boxes

def visualize_labels(data_dir, num_samples, split):
    """Main function to visualize YOLO labels"""
    print(f"Loading dataset configuration from {data_dir}")
    
    # Load dataset configuration
    try:
        config = load_dataset_config(data_dir)
        class_names = config.get('names', {})
        num_classes = config.get('nc', len(class_names))
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return
    
    print(f"Found {num_classes} classes: {list(class_names.values())}")
    
    # Generate colors for classes
    class_colors = get_class_colors(num_classes)
    
    # Create visualization output directory
    data_path = Path(data_dir)
    viz_dir = data_path / 'visualization'
    viz_dir.mkdir(exist_ok=True)
    
    # Get list of images in the specified split
    images_dir = data_path / 'images' / split
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} not found")
        return
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in {split} split")
    
    # Randomly sample images if we have more than requested
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    else:
        num_samples = len(image_files)
    
    print(f"Processing {num_samples} samples...")
    
    # Process each sample
    total_boxes = 0
    successful_samples = 0
    classes_found = set()
    
    for i, img_file in enumerate(image_files):
        print(f"Processing sample {i+1}/{num_samples}: {img_file.name}")
        
        # Load and visualize sample
        annotated_image, num_boxes = load_and_visualize_sample(
            data_dir, img_file.name, split, class_names, class_colors
        )
        
        if annotated_image is not None:
            # Save annotated image
            output_filename = f"{split}_{img_file.stem}_annotated.jpg"
            output_path = viz_dir / output_filename
            cv2.imwrite(str(output_path), annotated_image)
            
            successful_samples += 1
            total_boxes += num_boxes
            
            # Track which classes we found
            if num_boxes > 0:
                # Read labels to get class IDs
                label_filename = img_file.stem + '.txt'
                label_path = data_path / 'labels' / split / label_filename
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                classes_found.add(class_id)
            
            print(f"  - Saved {output_filename} with {num_boxes} bounding boxes")
        else:
            print(f"  - Failed to process {img_file.name}")
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Total images processed: {successful_samples}/{num_samples}")
    print(f"Total bounding boxes: {total_boxes}")
    
    if successful_samples > 0:
        avg_boxes = total_boxes / successful_samples
        print(f"Average boxes per image: {avg_boxes:.2f}")
    
    if classes_found:
        found_class_names = [class_names.get(cid, f"class_{cid}") for cid in sorted(classes_found)]
        print(f"Classes found: {', '.join(found_class_names)}")
    else:
        print("No classes found (all images may be background/good samples)")
    
    print(f"Visualization images saved to: {viz_dir}")
    
    return successful_samples, total_boxes, classes_found

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO format labels')
    parser.add_argument('--data_dir', required=True, 
                        help='Path to processed dataset directory (e.g., data/processed/bottle)')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--split', choices=['train', 'val'], default='val',
                        help='Dataset split to visualize (default: val)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO Label Visualizer")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {args.num_samples}")
    print("="*60)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Run visualization
    visualize_labels(args.data_dir, args.num_samples, args.split)

if __name__ == "__main__":
    main()