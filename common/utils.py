import os, shutil

def compose_filename(image_path: str, postfix: str, extension: str="png") -> str:
    """
    Given an image path like ./images/1. ARTWORK COLLISION/G6YK54W3653-MCDM.png,
    create its payload file name as artifacts/G6YK54W3653-MCDM_ROI.png
    """
    # Extract the filename from the path
    filename = os.path.basename(image_path)
    # Remove the extension
    base_name = os.path.splitext(filename)[0]
    # Create the payload filename with the full base name + postfix
    return f"artifacts/{base_name}_{postfix}.{extension}"


def copy_file(source_file: str, target_file: str) -> None:
    """
    Copies a file from source_file to target_file.
    Creates the target directory if it doesn't exist.
    
    Args:
        source_file: Path to the source file
        target_file: Path to the target file
    """
    # Create target directory if it doesn't exist
    target_dir = os.path.dirname(target_file)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    shutil.copy2(source_file, target_file)