
# %%
import pandas as pd

# Read the CSV file
df = pd.read_csv('grid_points.csv')

# Filter the points by Shore equal to "Onshore"
filtered_df = df[df['Shore'] == 'Onshore']

# Store the filtered points in a new CSV file
filtered_df.to_csv('grid_points_onshore.csv', index=False)

# Print the first 5 rows of the filtered DataFrame
print(filtered_df.head())

# %%
