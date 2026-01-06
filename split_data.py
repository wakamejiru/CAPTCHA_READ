import os
import shutil
import random
from pathlib import Path

def split_data():
    # Define directories
    base_dir = Path(os.getcwd())
    source_dir = base_dir / "archive (1)"
    test_dir = base_dir / "test"
    study_dir = base_dir / "study"

    # Ensure source exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create target directories if they don't exist
    test_dir.mkdir(exist_ok=True)
    study_dir.mkdir(exist_ok=True)

    # Allowed extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # Get all image files
    print("Scanning files...")
    all_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    total_files = len(all_files)
    if total_files == 0:
        print("No image files found in source directory.")
        return

    print(f"Found {total_files} images.")

    # Shuffle the list
    random.shuffle(all_files)

    # Calculate split index
    split_index = int(total_files * 0.2)
    
    test_files = all_files[:split_index]
    study_files = all_files[split_index:]

    print(f"Moving {len(test_files)} files to 'test'...")
    for file_path in test_files:
        shutil.move(str(file_path), str(test_dir / file_path.name))

    print(f"Moving {len(study_files)} files to 'study'...")
    for file_path in study_files:
        shutil.move(str(file_path), str(study_dir / file_path.name))

    print("Data split complete.")
    print(f"Test: {len(test_files)}")
    print(f"Study: {len(study_files)}")

if __name__ == "__main__":
    split_data()
