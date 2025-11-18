import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import csv
import os
import re

use_log_scale = False
max_val_filter = None
num_bins = 100

sortCsv = False

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]
filename_stripped = os.path.basename(filename)
m = re.search(r"-d(?P<d>\d+)n(?P<n>\d+)k(?P<k>\d+)trials(?P<trials>\d+)\b", filename_stripped)
the_d = "unknown"
the_n = "unknown"
the_k = "unknown"
the_bin_size = "unknown"
if m:
    the_d = m["d"]
    the_n = m["n"]
    the_k = m["k"]
    the_bin_size = str(pow(int(the_d), 2 * int(the_n)) - int(the_k))
print(f"d {the_d}, n {the_n}, k {the_k}, Bin size is {the_bin_size}")

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


vec_accept_count = df['VectorsAcceptedCount']
# Convert microseconds to milliseconds
durations = df['TimeFor_u_FilteringInMicroS'] / 1000.0

# Compute statistics
count = vec_accept_count.shape[0]
mean_val = vec_accept_count.mean()
std_val = vec_accept_count.std()
min_val = vec_accept_count.min()
max_val = vec_accept_count.max()
percentiles = np.percentile(vec_accept_count, [25, 50, 75, 90, 95, 99])

title_suffix = ""

# Print statistics
print(f"Statistics for 'VectorsAcceptedCount' from file, scaled to milliseconds: {filename}{title_suffix}")
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


outliers = df[df['VectorsAcceptedCount'] > percentiles[5]]
outliers = outliers.sort_values(by=['VectorsAcceptedCount'], ascending=True, inplace=False)
print(f"{outliers.shape[0]} outliers in unfiltered dataset (> 99th percentile)")
print(outliers)
if outliers.shape[0] > 0:
    basename_noext = os.path.splitext(os.path.basename(filename))[0]
    outliers.to_csv(f"{basename_noext}-outliers.csv", index=False)
    print(f"Wrote to {basename_noext}-outliers.csv")

if use_log_scale:
    bins = np.logspace(np.log10(vec_accept_count.min()), np.log10(vec_accept_count.max()), num_bins)
else:
    bins = np.linspace(vec_accept_count.min(), vec_accept_count.max(), num_bins)

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(vec_accept_count, bins=bins, edgecolor='black')


if use_log_scale:
    plt.xscale('log')
    plt.xlabel('Number of vectors accepted (log scale)')
    plt.xticks([1,2,4,8], ['1', '2', '4', '8'])
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    #ax.xaxis.set_major_locator(LogLocator(base=3.0, subs=(1.0,), numticks=10))
    #ax.xaxis.set_minor_locator(LogLocator(base=3.0, subs=(1.0,), numticks=10))
    #ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))

else:
    plt.xlabel('Number of vectors accepted')
plt.ylabel('Frequency')
plt.minorticks_on()

plt.title(f'Histogram of necessary-condition accepted vectors on random bins (d={the_d}, n={the_n}, k={the_k}, bin size={the_bin_size})')
plt.grid(True, linestyle='--', alpha=0.6)

image_name = f"{os.path.basename(filename)}.png"
plt.savefig(image_name, dpi=300)

plt.close()
