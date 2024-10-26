import numpy as np
import pandas as pd

# Generate a random dataset of RGB colors
np.random.seed(42)  # For reproducibility
num_samples = 5000  # Number of samples

# Generate random RGB values
colors = np.random.randint(0, 256, size=(num_samples, 3))

# Create a label: 1 if the color is predominantly red, 0 otherwise
# A simple heuristic: if the red value is greater than both green and blue values
labels = np.where((colors[:, 0] > colors[:, 1]) & (colors[:, 0] > colors[:, 2]), 1, 0)

# Create a DataFrame for better visualization
df = pd.DataFrame(colors, columns=['Red', 'Green', 'Blue'])
df['Is_Red'] = labels

# Display the dataset
# import ace_tools as tools; tools.display_dataframe_to_user(name="RGB Color Dataset", dataframe=df)

df.head()
print(df)
df.to_csv("./color_dataset.csv", index=False)