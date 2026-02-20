import requests
import os


url = "https://raw.githubusercontent.com/GIST-CSBL/AMP-BERT/main/all_veltri.csv"

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

# Where the file will be saved
output_path = "data/all_veltri.csv"

# Download file
response = requests.get(url)

with open(output_path, "wb") as f:
    f.write(response.content)

print("Veltri dataset downloaded successfully!")
