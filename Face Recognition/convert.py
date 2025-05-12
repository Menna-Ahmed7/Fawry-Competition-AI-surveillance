import pandas as pd
import json
import csv

# Read the original CSV file (submission.csv)
df = pd.read_csv("submission.csv")  # Update the file path if necessary

new_rows = []

# First, add 429 new tracking rows
for i in range(429):
    new_rows.append([i, i+1, '[]', 'tracking'])

# Then process each row from submission.csv and offset the ID by 429
for index, row in df.iterrows():
    new_id = index + 429  # So that submission rows start at row 430
    gt_value = row["gt"]
    image_value = row["image"].replace("test/", "test_set/")  # Adjust image path as needed
    data_dict = {"gt": gt_value, "image": image_value}
    new_rows.append([new_id, -1, json.dumps(data_dict), "face_reid"])

# Create a new DataFrame with the desired column names
new_df = pd.DataFrame(new_rows, columns=["ID", "frame", "objects", "objective"])

# Write the new DataFrame to a CSV file with comma-separated columns.
# The csv.QUOTE_MINIMAL option ensures fields with commas (like the JSON string) are quoted.
new_df.to_csv("output.csv", index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
