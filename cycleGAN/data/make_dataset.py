from zipfile import ZipFile
import os

# Extract the data
DATA_DIR = "data"
EXTRACT_PATH = "data"

TRAIN_DIR = os.path.join(DATA_DIR, "Assignment_4/Train")
TRAIN_FILES = os.listdir(TRAIN_DIR)

TEST_DIR = os.path.join(DATA_DIR, "Assignment_4/Test")
TEST_FILES = os.listdir(os.path.join(DATA_DIR, "Assignment_4/Test"))

def extract_zipfile(file_path, extract_path):
    with ZipFile(file_path, 'r') as asz:
        asz.extractall(extract_path)

extract_zipfile(os.path.join(DATA_DIR, "Assignment_4-20240408T115837Z-002.zip"), EXTRACT_PATH)

for file in TRAIN_FILES:
    if file.endswith(".zip"):
        file_path = os.path.join(TRAIN_DIR, file)
        extract_zipfile(file_path, TRAIN_DIR)

for file in TEST_FILES:
    if file.endswith(".zip"):
        file_path = os.path.join(TEST_DIR, file)
        extract_zipfile(file_path, TEST_DIR)
