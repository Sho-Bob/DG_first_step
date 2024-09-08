import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/Users/s_wada/Documents/Stanford_Research/Research Meeting/0909/Linear_advection_L2norm.xlsx'
data = pd.read_excel(file_path)
# data = pd.read_excel(file_path, sheet_name=1)

# Extract data from the first three columns
x = data.iloc[:, 0]  # First column for x-axis
y_points = data.iloc[:, 1]  # Second column for points
y_line = data.iloc[:, 2]  # Third column for line

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y_points, color='b', label='L2 norm', marker='o')  # Plot points
plt.plot(x, y_line, color='r', label='4th order')  # Plot line

# Set logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Label the axes
plt.xlabel('N (Number of elements)')
plt.ylabel('Error')

# Add title and legend
plt.title('Logarithmic Plot of Data')
plt.legend()

# Show grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()