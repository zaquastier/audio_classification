import requests
import tarfile
import os

# URL of the dataset
url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"

# Download the dataset
# response = requests.get(url, stream=True)
# with open("urban8k.tgz", "wb") as file:
#     for chunk in response.iter_content(chunk_size=1024):
#         if chunk:
#             file.write(chunk)

# Extract the tarball
with tarfile.open("urban8k.tgz", "r:gz") as tar:
    tar.extractall()

# Remove the downloaded tarball
os.remove("urban8k.tgz")

print("Dataset downloaded and extracted successfully!")