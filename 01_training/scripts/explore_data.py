import os
from pathlib import Path
from collections import defaultdict

def explore_mvtec_data(data_dir):
    """
    Explore MVTec Anomaly Detection dataset structure and count images.
    """
    data_path = Path(data_dir)
    categories = {}
    
    # Find all category directories (exclude files like license.txt, readme.txt)
    for category_dir in data_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            categories[category_name] = {
                'train_good': 0,
                'test_good': 0,
                'test_defects': defaultdict(int),
                'total': 0
            }
            
            # Check train directory
            train_dir = category_dir / 'train'
            if train_dir.exists():
                train_good_dir = train_dir / 'good'
                if train_good_dir.exists():
                    categories[category_name]['train_good'] = len(list(train_good_dir.glob('*.png')))
            
            # Check test directory
            test_dir = category_dir / 'test'
            if test_dir.exists():
                for defect_type_dir in test_dir.iterdir():
                    if defect_type_dir.is_dir():
                        defect_type = defect_type_dir.name
                        image_count = len(list(defect_type_dir.glob('*.png')))
                        
                        if defect_type == 'good':
                            categories[category_name]['test_good'] = image_count
                        else:
                            categories[category_name]['test_defects'][defect_type] = image_count
            
            # Calculate total
            total = categories[category_name]['train_good'] + categories[category_name]['test_good']
            for count in categories[category_name]['test_defects'].values():
                total += count
            categories[category_name]['total'] = total
    
    return categories

def print_category_details(categories):
    """Print detailed information for each category."""
    print("=" * 80)
    print("MVTec Anomaly Detection Dataset Analysis")
    print("=" * 80)
    
    for category, data in sorted(categories.items()):
        print(f"\n📁 Category: {category.upper()}")
        print(f"   Training images (good): {data['train_good']}")
        print(f"   Test images (good): {data['test_good']}")
        
        if data['test_defects']:
            print("   Test images by defect type:")
            for defect_type, count in sorted(data['test_defects'].items()):
                print(f"     - {defect_type}: {count}")
        
        print(f"   Total images: {data['total']}")

def print_summary_table(categories):
    """Print summary table of all categories."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Category':<15} {'Train':<8} {'Test Good':<10} {'Test Defects':<13} {'Total':<8}")
    print("-" * 80)
    
    total_train = 0
    total_test_good = 0
    total_test_defects = 0
    total_images = 0
    
    for category, data in sorted(categories.items()):
        test_defects_count = sum(data['test_defects'].values())
        
        print(f"{category:<15} {data['train_good']:<8} {data['test_good']:<10} "
              f"{test_defects_count:<13} {data['total']:<8}")
        
        total_train += data['train_good']
        total_test_good += data['test_good']
        total_test_defects += test_defects_count
        total_images += data['total']
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {total_train:<8} {total_test_good:<10} "
          f"{total_test_defects:<13} {total_images:<8}")
    
    print(f"\nDataset Statistics:")
    print(f"  - Number of categories: {len(categories)}")
    print(f"  - Total images: {total_images:,}")
    print(f"  - Training images: {total_train:,}")
    print(f"  - Test images: {total_test_good + total_test_defects:,}")
    print(f"    * Good test images: {total_test_good:,}")
    print(f"    * Anomaly test images: {total_test_defects:,}")

if __name__ == "__main__":
    # Path to the raw data directory
    data_directory = "/Users/geseuteu/vision-inspection-portfolio/01_training/data/raw"
    
    print("Scanning MVTec dataset...")
    categories = explore_mvtec_data(data_directory)
    
    if not categories:
        print("No categories found in the specified directory.")
    else:
        print_category_details(categories)
        print_summary_table(categories)