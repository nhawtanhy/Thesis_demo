import json

# Input file
json_path = r"C:\Users\nhata\Desktop\Thesis_All\Data\D_Contrast\depAPI_contrastive_deepseek.json"

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect unique mappings
unique_pairs = set()

for item in data:
    deprecated = item.get("deprecated api", [])
    replacement = item.get("replacement api", "")

    # Handle both list and string formats
    if isinstance(deprecated, list):
        for dep in deprecated:
            unique_pairs.add((dep, replacement))
    else:
        unique_pairs.add((deprecated, replacement))

# Sort alphabetically
unique_pairs = sorted(unique_pairs)

print(f"Total unique mappings: {len(unique_pairs)}\n")

for dep, rep in unique_pairs:
    print(f"{dep}  -->  {rep}")