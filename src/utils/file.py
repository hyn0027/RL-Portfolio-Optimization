import os


def create_path_recursively(path: str) -> None:
    """Recursively create a path if it doesn't exist.

    Args:
        path (str): the path to create
    """
    if not os.path.exists(path):
        # Split the path into the current directory and the base name
        head, _ = os.path.split(path)
        # Recursively create the parent directory
        create_path_recursively(head)
        # Create the current directory
        os.mkdir(path)
