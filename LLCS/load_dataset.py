import numpy as np
import pandas as pd

df = pd.read_csv("./dataset/color_dataset.csv")
# Initialize a list to store the results
processed_data = []

# Loop through each row in the CSV
for index, row in df.iterrows():
    rgb_values = row[['Red', 'Green', 'Blue']].to_numpy()
    label = row['Is_Red']
    
    # Create the output array based on the label
    if label == 0:
        output = np.array([0, 1])
    else:
        output = np.array([1, 0])
    
    # Append the results
    processed_data.append((rgb_values, output))

# Display the first 5 processed rows as an example
print(processed_data[:5])

