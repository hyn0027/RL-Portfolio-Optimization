import os


def create_path_recursively(path):
    """
    Recursively create a path if it doesn't exist.
    """
    if not os.path.exists(path):
        # Split the path into the current directory and the base name
        head, tail = os.path.split(path)
        # Recursively create the parent directory
        create_path_recursively(head)
        # Create the current directory
        os.mkdir(path)
