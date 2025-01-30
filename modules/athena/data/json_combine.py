import json

json_files = ["kasparov_training_data.json", "carlsen_training_data.json", "fischer_training_data.json"]

combined_data = []
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        combined_data.extend(data)

with open("training_data.json", "w") as f:
    json.dump(combined_data, f, indent=4)

print("Combined training data saved to combined_training_data.json")
