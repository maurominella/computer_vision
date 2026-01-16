import os, shutil

def compose_filename(image_path: str, postfix: str, extension: str="png") -> str:
    """
    Given an image path like ./images/1. ARTWORK COLLISION/J74Q10KAUG0-G6N3.png,
    create its payload file name as artifacts/J74Q10KAUG0-ROI.json
    """
    # Extract the filename from the path
    filename = os.path.basename(image_path)
    # Remove the extension
    base_name = os.path.splitext(filename)[0]
    # Split by '-' and take the first part (e.g., J74Q10KAUG0 from J74Q10KAUG0-G6N3)
    prefix = base_name.split('-')[0]
    # Create the payload filename
    return f"artifacts/{prefix}-{postfix}.{extension}"


def copy_file(source_path: str, target_path: str) -> None:
    """
    Copies a file from source_path to target_path.
    Creates the target directory if it doesn't exist.
    
    Args:
        source_path: Path to the source file
        target_path: Path to the target file
    """
    # Create target directory if it doesn't exist
    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    shutil.copy2(source_path, target_path)