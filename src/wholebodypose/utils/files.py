"""File system utilities for directory and file operations."""

from pathlib import Path


def create_directory(path: str | Path) -> Path:
    """
    Create all directories in the given path if they don't exist.

    Args:
        path: Directory path to create

    Returns:
        Path object of the created directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def find_files_by_extension(folder_path: str | Path, extension: str = ".mp4", recursive: bool = True) -> list[str]:
    """
    Search for all files with specified extension.

    Args:
        folder_path: Root directory to search
        extension: File extension (e.g., '.mp4', '.avi')
        recursive: If True, searches subdirectories

    Returns:
        List of absolute file paths
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    if not extension.startswith("."):
        extension = f".{extension}"

    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    files = [str(f.resolve()) for f in folder.glob(pattern) if f.is_file()]

    return files
