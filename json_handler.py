import json
import os

class JsonHandler:
    """
    Handles loading and saving data to JSON files.
    """
    def __init__(self, directory='./'): # changed default to current directory
        """
        Initializes the JsonHandler with a directory.
        """
        self.directory = directory
        if not os.path.exists(self.directory):
            print(f'It appears that the directory {self.directory} is empty. Creating it')
            os.makedirs(self.directory)

    def save_data(self, data, filename):
        """
        Saves data to a JSON file.

        Args:
            data (dict): The data to save.  Must be serializable to JSON.
            filename (str): The name of the file (without extension).
        """
        filepath = os.path.join(self.directory, f'{filename}.json')
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)  # Pretty printing
                f.close()
        except Exception as e:
            print(f"Error saving to {filepath}: {e}")
            raise  # Re-raise the exception to be handled by caller

    def load_data(self, filename):
        """
        Loads data from a JSON file.

        Args:
            filename (str): The name of the file (without extension).

        Returns:
            dict: The loaded data, or None if the file does not exist or cannot be loaded.
        """
        filepath = os.path.join(self.directory, f'{filename}.json')
        if not os.path.exists(filepath):
            print(f"Warning: File not found at {filepath}")
            return None
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading from {filepath}: {e}")
            raise #re-raise
