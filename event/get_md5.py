import hashlib
import os

# The path where your valid dataset files are located
# Based on your logs, this is the directory:
data_dir = '/data/rboone/datasets/c10_dvs_temp/download'

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

print(f"Scanning directory: {data_dir}\n")
print("-" * 60)
print(f"{'Filename':<20} | {'MD5 Hash'}")
print("-" * 60)

# The expected files for CIFAR10-DVS are usually these zip files
expected_files = [
    "airplane.zip", "automobile.zip", "bird.zip", "cat.zip", "deer.zip",
    "dog.zip", "frog.zip", "horse.zip", "ship.zip", "truck.zip"
]

for filename in expected_files:
    file_path = os.path.join(data_dir, filename)
    
    if os.path.exists(file_path):
        md5_hash = calculate_md5(file_path)
        print(f"'{filename}', '{md5_hash}'")
    else:
        print(f"MISSING: {filename}")

print("-" * 60)