import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/Users/s_wada/Documents/Stanford_Research/Research Meeting/0916/Linear_advection_L2norm.xlsx'
data = pd.read_excel(file_path)
# data = pd.read_excel(file_path, sheet_name=1)

# Extract data from the first three columns
x = data.iloc[:, 0]  # First column for x-axis
L2norm4th = data.iloc[:, 1]  # Second column for points
fouth_o = data.iloc[:, 2]  # Third column for line
fifth_o = data.iloc[:,4]

L2norm3rd = data.iloc[:,5]
L2norm2nd = data.iloc[:,6]
fourth_o2 = data.iloc[:,7]
thrid_o2 = data.iloc[:,8]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x, L2norm4th, color='b', label='p4', marker='o')  # Plot points
# plt.plot(x, fouth_o, color='r', label='4th order')  # Plot line
plt.plot(x, fifth_o, color='r', label='5th order')  # Plot line
# Set logarithmic scale
plt.xscale('log')
plt.yscale('log')
# Label the axes
plt.xlabel('N (Number of elements)')
plt.ylabel('L2 norm')
# Add title and legend
# plt.title('Logarithmic Plot of Data')
plt.legend()
# Show grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Display the plot
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(x, L2norm3rd, color='b', label='p3', marker='o')  # Plot points
plt.plot(x, fourth_o2, color='r', label='4th order')  # Plot line
# Set logarithmic scale
plt.xscale('log')
plt.yscale('log')
# Label the axes
plt.xlabel('N (Number of elements)')
plt.ylabel('L2 norm')
# Add title and legend
# plt.title('Logarithmic Plot of Data')
plt.legend()
# Show grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Display the plot
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x, L2norm2nd, color='b', label='p2', marker='o')  # Plot points
plt.plot(x, thrid_o2, color='r', label='3rd order')  # Plot line
# Set logarithmic scale
plt.xscale('log')
plt.yscale('log')
# Label the axes
plt.xlabel('N (Number of elements)')
plt.ylabel('L2 norm')
# Add title and legend
# plt.title('Logarithmic Plot of Data')
plt.legend()
# Show grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Display the plot
plt.show()

