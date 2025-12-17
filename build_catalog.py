import pandas as pd
import json
import os
from urllib.parse import urlparse

DATASET_PATH = "Gen_AI Dataset.xlsx"
OUTPUT_PATH = "data/catalog.json"

os.makedirs("data", exist_ok=True)

def infer_name(url):
    path = urlparse(url).path
    slug = path.split("/")[-1]
    if not slug:
        slug = path.split("/")[-2]
    return slug.replace("-", " ").title()

def infer_type(name):
    name = name.lower()
    if any(k in name for k in ["personality", "behavior", "behaviour"]):
        return "Personality & Behavior"
    return "Knowledge & Skills"

df = pd.read_excel(DATASET_PATH, sheet_name="Train-Set")
df.columns = [c.lower() for c in df.columns]

urls = df["assessment_url"].unique()

catalog = []
for url in urls:
    name = infer_name(url)
    t = infer_type(name)
    catalog.append({
        "name": name,
        "url": url,
        "description": f"{name} assessment",
        "test_type": t
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(catalog, f, indent=2)

print("catalog.json created")
