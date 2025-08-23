import sys
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]

# Load CSV file
#df = pd.read_csv(filename)
#durations_micros = df['DurationMicroS']
#durations = durations_micros / 1000

conn = sqlite3.connect(filename)
# Read data into a pandas DataFrame
df = pd.read_sql_query("SELECT DurationMicroS FROM results", conn)
conn.close()

# Convert microseconds to milliseconds
durations = df['DurationMicroS'] / 1000.0

# Compute statistics
mean_val = durations.mean()
std_val = durations.std()
min_val = durations.min()
max_val = durations.max()
percentiles = np.percentile(durations, [25, 50, 75, 90, 95, 99])

# Print statistics
print(f"Statistics for 'DurationMicroS' from file, scaled to milliseconds: {filename}")
print(f"Mean: {mean_val:.3f}")
print(f"Std Deviation: {std_val:.3f}")
print(f"Min: {min_val}")
print(f"Max: {max_val}")
print(f"25th percentile: {percentiles[0]}")
print(f"50th percentile (median): {percentiles[1]}")
print(f"75th percentile: {percentiles[2]}")
print(f"90th percentile: {percentiles[3]}")
print(f"95th percentile: {percentiles[4]}")
print(f"99th percentile: {percentiles[5]}")

bins = np.logspace(np.log10(durations.min()), np.log10(durations.max()), 50) # 50
# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(durations, bins=bins, edgecolor='black')
plt.xscale('log')

plt.xlabel('Duration (milliseconds, log scale)')
plt.ylabel('Frequency')
plt.title('Histogram for Clifford-conjugate test on all Clifford (d=3, n=2)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('histogram-d3n2-fromresults.png', dpi=300)

plt.close()
