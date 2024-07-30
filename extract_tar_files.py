import os
import tarfile


def extract_tar_gz(file_path, extract_path='.'):
    """
    Extracts a .tar.gz file to the specified directory.

    :param file_path: Path to the .tar.gz file
    :param extract_path: Directory where the files will be extracted (default is
    current directory)
    """
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print(f"Extracted {file_path} to {extract_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_and_delete_tar_gz(file_path, extract_path='.'):
    """
    Extracts a .tar.gz file and then deletes it.

    :param file_path: Path to the .tar.gz file
    :param extract_path: Directory where the files will be extracted (default is
    current directory)
    """
    extract_tar_gz(file_path, extract_path)
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"An error occurred while deleting {file_path}: {e}")


def extract_all_tar_gz_in_directory(directory_path, extract_path='.'):
    """
    Iterates over all .tar.gz files in a directory, extracts them, and then
    deletes them.

    :param directory_path: Path to the directory containing .tar.gz files
    :param extract_path: Directory where the files will be extracted (default is
    current directory)
    """
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.tar.gz'):
            file_path = os.path.join(directory_path, file_name)
            extract_and_delete_tar_gz(file_path, extract_path)


# Example usage:
source_dir = os.path.join(os.getcwd(), 'data', 'EEG')
extract_all_tar_gz_in_directory(source_dir, source_dir)
