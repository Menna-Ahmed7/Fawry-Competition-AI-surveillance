import pandas as pd

# Define file paths (adjust if necessary)
submission_file_path = r'face only.csv'
tracking_file_path = r'd:\GItHub Reops\yolov11\tracking_output_osnet_75_msmt177_1_111500.csv'
output_file_path = 'Face & Tracking.csv'

# Load both CSV files into DataFrames
submission_df = pd.read_csv(submission_file_path)
tracking_df = pd.read_csv(tracking_file_path)

# Check that tracking_df has at least 429 rows
if len(tracking_df) < 429:
    raise ValueError("Tracking output file has less than 429 rows!")

# Overwrite the first 429 rows of the submission file with those from the tracking output
submission_df.iloc[:429] = tracking_df.iloc[:429]

# Save the updated DataFrame to a new CSV file
submission_df.to_csv(output_file_path, index=False)
print(f"Updated submission file saved to: {output_file_path}")
