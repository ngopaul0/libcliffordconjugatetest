import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

use_log_scale = False
max_val_filter = None
num_bins = 100

sortCsv = False

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]

if sortCsv:
    # Read CSV and sort by CliffordIndex
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        sorted_rows = sorted(reader, key=lambda row: row[0]) 
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)   
        writer.writerows(sorted_rows)

# Load CSV file
df = pd.read_csv(filename)

duplicates_exist = df['CliffordIndex'].duplicated().any()

# Convert microseconds to milliseconds
durations = df['DurationMicroS'] / 1000.0

if max_val_filter:
    durations = durations[df['DurationMicroS'] / 1000.0 < max_val_filter]

# Compute statistics
count = durations.shape[0]
mean_val = durations.mean()
std_val = durations.std()
min_val = durations.min()
max_val = durations.max()
percentiles = np.percentile(durations, [25, 50, 75, 90, 95, 99])

title_suffix = ""
if max_val_filter:
    title_suffix = f" (filtered <= {max_val_filter})"

# Print statistics
print(f"Statistics for 'DurationMicroS' from file, scaled to milliseconds: {filename}{title_suffix}")
print(f"Duplicates: {duplicates_exist}")
if duplicates_exist:
    duplicate_values = df['CliffordIndex'][df['CliffordIndex'].duplicated()].unique()
    print(duplicate_values)
print(f"Count: {count}")
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


outliers = df[df['DurationMicroS'] / 1000.0 > percentiles[5]]
outliers = outliers.sort_values(by=['DurationMicroS'], ascending=True, inplace=False)
print(f"{outliers.shape[0]} outliers in unfiltered dataset (> 99th percentile)")
print(outliers)
if outliers.shape[0] > 0:
    basename_noext = os.path.splitext(os.path.basename(filename))[0]
    outliers.to_csv(f"{basename_noext}-outliers.csv", index=False)
    print(f"Wrote to {basename_noext}-outliers.csv")

if use_log_scale:
    bins = np.logspace(np.log10(durations.min()), np.log10(durations.max()), num_bins)
else:
    bins = np.linspace(durations.min(), durations.max(), num_bins)

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(durations, bins=bins, edgecolor='black')
if use_log_scale:
    plt.xscale('log')
    plt.xlabel('Duration (milliseconds, log scale)')
else:
    plt.xlabel('Duration (milliseconds)')
plt.ylabel('Frequency')

plt.title(f'Histogram for Clifford-conjugate test on M = W(p,q) for all Clifford (d=3, n=2){title_suffix}')
plt.grid(True, linestyle='--', alpha=0.6)

image_name = f"{os.path.basename(filename)}.png"
plt.savefig(image_name, dpi=300)

plt.close()
