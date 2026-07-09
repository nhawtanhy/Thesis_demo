import json

file_path = r"C:\Users\nhata\Downloads\Thesis_data\RAG\D_test.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each dictionary to a hashable string
unique_instances = {json.dumps(item, sort_keys=True) for item in data}

print(f"Total instances: {len(data)}")
print(f"Unique instances: {len(unique_instances)}")