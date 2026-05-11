import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_data():
    # 1. Reset: Move everything from val and test back to train
    print("Consolidating all images into training folder for a fresh split...")
    for target_dir in [VAL_DIR, TEST_DIR]:
        if not target_dir.exists():
            continue
        for category in os.listdir(target_dir):
            cat_source_path = target_dir / category
            cat_train_path = TRAIN_DIR / category
            if not cat_source_path.is_dir(): continue
            
            for img in os.listdir(cat_source_path):
                shutil.move(str(cat_source_path / img), str(cat_train_path / img))

    # 2. Perform the 70/15/15 split
    print(f"Splitting data into Train ({int(TRAIN_RATIO*100)}%), Val ({int(VAL_RATIO*100)}%), and Test ({int(TEST_RATIO*100)}%)...")
    
    for category in os.listdir(TRAIN_DIR):
        cat_train_path = TRAIN_DIR / category
        cat_val_path = VAL_DIR / category
        cat_test_path = TEST_DIR / category
        
        if not cat_train_path.is_dir(): continue
        os.makedirs(cat_val_path, exist_ok=True)
        os.makedirs(cat_test_path, exist_ok=True)
        
        # Get images and shuffle with fixed seed
        images = os.listdir(cat_train_path)
        random.seed(42) 
        random.shuffle(images)
        
        # Calculate indices
        total_count = len(images)
        val_end = int(total_count * VAL_RATIO)
        test_end = val_end + int(total_count * TEST_RATIO)
        
        val_images = images[:val_end]
        test_images = images[val_end:test_end]
        
        # Move to Val
        for img in val_images:
            shutil.move(str(cat_train_path / img), str(cat_val_path / img))
            
        # Move to Test
        for img in test_images:
            shutil.move(str(cat_train_path / img), str(cat_test_path / img))
        
        print(f"  - {category}: {total_count - len(val_images) - len(test_images)} Train | {len(val_images)} Val | {len(test_images)} Test")

    print("\nDone! Data is now split 70/15/15.")

if __name__ == "__main__":
    split_data()