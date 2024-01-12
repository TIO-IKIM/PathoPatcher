import unittest
import pandas as pd
from pathlib import Path
from pathopatch.utils.file_handling import load_wsi_files_from_csv


class TestLoadWSIFilesFromCSV(unittest.TestCase):
    def setUp(self):
        """Set up a temporary CSV file for testing."""
        self.test_csv_path = Path("test.csv")
        self.data = {
            "Filename": [
                "file1.svs",
                "file2.svs",
                "file3.jpg",
                "file4.svs",
                "file5.png",
            ]
        }
        self.df = pd.DataFrame(self.data)
        self.df.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Remove the temporary CSV file after testing."""
        self.test_csv_path.unlink()

    def test_load_wsi_files_from_csv_valid(self):
        """Test load_wsi_files_from_csv with valid inputs."""
        expected_output = ["file1.svs", "file2.svs", "file4.svs"]
        actual_output = load_wsi_files_from_csv(self.test_csv_path, "svs")
        self.assertEqual(actual_output, expected_output)

    def test_load_wsi_files_from_csv_invalid_extension(self):
        """Test load_wsi_files_from_csv with an invalid file extension."""
        expected_output = []
        actual_output = load_wsi_files_from_csv(self.test_csv_path, "invalid_extension")
        self.assertEqual(actual_output, expected_output)

    def test_load_wsi_files_from_csv_nonexistent_file(self):
        """Test load_wsi_files_from_csv with a nonexistent CSV file."""
        with self.assertRaises(FileNotFoundError):
            load_wsi_files_from_csv("nonexistent.csv", "svs")


if __name__ == "__main__":
    unittest.main()
