import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file from the second sheet
file_path = '/Users/s_wada/Documents/Stanford_Research/Research Meeting/0909/Linear_advection_L2norm.xlsx'
data = pd.read_excel(file_path, sheet_name=1)  # Load the second sheet (index 1)

# Extract data from the first and second columns
x = data.iloc[:, 0]  # First column for x-axis
y = data.iloc[:, 1]  # Second column for y-axis

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o-', label='Data Points')  # Plot points

# Set logarithmic scale for both axes
# plt.xscale('log')
plt.yscale('log')

# Label the axes
plt.xlabel('Approximation order')
plt.ylabel('L2norm')

# Add title and legend
plt.title('p-refinement')
plt.legend()

# Show grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()