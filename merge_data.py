import pandas as pd

# List of your specific gesture CSV filenames
csv_files = [
    "gesture_data_hi.csv",
    "gesture_data_thankyou.csv",
    "gesture_data_bye.csv",
    "gesture_data_yes.csv",
    "gesture_data_no.csv",
    "gesture_data_i love you.csv"
]

dataframes = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded: {file} - {len(df)} samples")
        dataframes.append(df)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

# Merge all DataFrames into one
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv("gesture_data_all.csv", index=False)
    print(f"\nAll files merged successfully into gesture_data_all.csv")
    print(f"Total samples: {len(merged_df)}")
else:
    print("\nNo data to merge. Please check your CSV file paths.")
