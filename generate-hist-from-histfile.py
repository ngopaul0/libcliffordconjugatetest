import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df_unfiltered = pd.read_csv('histogram-d3n2.csv')

# Filter bins with nonzero counts
nonzero_bins = df_unfiltered[df_unfiltered['Count'] > 0]

# Find the maximum bin by upper edge
max_upper = 5000 #nonzero_bins['Upper'].max() + 500

# Keep only bins where Lower < max_upper (i.e., bins up to that max bin)
df = df_unfiltered[df_unfiltered['Lower'] < max_upper]

# Calculate bin centers and widths
bin_centers = (df['Lower'] + df['Upper']) / 2
bin_widths = df['Upper'] - df['Lower']

# Create the plot
plt.figure(figsize=(10,6))
plt.bar(bin_centers, df['Count'], width=bin_widths, edgecolor='black', align='center')

plt.xlabel('Runtime (bins of 50 ms)')
plt.ylabel('Count')
plt.title('Histogram for Clifford-conjugate test on all Clifford (d=3, n=2)')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot as a PNG file
plt.savefig('histogram-d3n2.png', dpi=300)

plt.close()