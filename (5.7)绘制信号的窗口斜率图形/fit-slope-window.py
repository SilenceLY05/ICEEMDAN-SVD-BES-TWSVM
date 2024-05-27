import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Constants
DIRECTORY_PATH = r'G:\毕业论文文件\OTDR_Data\新建文件夹2'  # Replace with the path of the folder containing your files
WINDOW_SIZE = 50
STEP_SIZE = 20  # New step size for sliding the window


# Get the latest file from the folder
def get_latest_file(directory):
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files if file.endswith('.txt') or file.endswith('.csv')]
    return max(paths, key=os.path.getctime) if paths else None


# Read data from the latest file
latest_file = get_latest_file(DIRECTORY_PATH)
if not latest_file:
    print("No data file found in the specified directory.")
    exit()

# Assuming the data file is structured as required (two columns separated by a comma)
data = pd.read_csv(latest_file, header=None)
data.columns = ['X', 'Y']

# Calculate the slope of the best-fit line within each window
slopes = []
positions = []

for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):  # Change the step from WINDOW_SIZE to STEP_SIZE
    end = start + WINDOW_SIZE
    window_data = data.iloc[start:end]

    # Fit a linear regression model
    X = window_data['X'].values.reshape(-1, 1)
    Y = window_data['Y'].values
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]

    # Store the slope and the midpoint position of the window
    slopes.append(slope)
    positions.append(window_data['X'].mean())

# Plot the results as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(positions, slopes, width=0.1, color='black')
plt.xlabel('Position (X)')
plt.ylabel('Slope')
plt.title('Slope of Best-Fit Line in Each Window')
plt.grid(True)
plt.show()
