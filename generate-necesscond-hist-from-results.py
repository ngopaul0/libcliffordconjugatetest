import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
from typing import Optional
import csv
import os
import re

use_log_scale = False
max_val_filter = None
start_hist_from_min = False
hist_max: int | None = None
num_bins = 30

sortCsv = False

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]
filename_stripped = os.path.basename(filename)
m = re.search(r"-d(?P<d>\d+)n(?P<n>\d+)k(?P<k>\d+)trials(?P<trials>\d+)(?P<bf>bf)?\b", filename_stripped)
the_d = "unknown"
the_n = "unknown"
the_k = "unknown"
the_vec_bin_size = None
is_basis_finding = False
if m:
    the_d = m["d"]
    the_n = m["n"]
    the_k = m["k"]
    the_vec_bin_size = pow(int(the_d), 2 * int(the_n)) - int(the_k)
    is_basis_finding = m.group("bf") is not None
print(f"d {the_d}, n {the_n}, k {the_k}, Bin size is {the_vec_bin_size}, basis finding {is_basis_finding}")


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
percentile_markers = [1, 25, 50, 75, 90, 95, 99]
percentiles = np.percentile(vec_accept_count, percentile_markers)

title_suffix = ""

# Print statistics
stat_string = f"Statistics for 'VectorsAcceptedCount' from file: {filename}{title_suffix}\n"
stat_string += f"Count: {count}\n"
stat_string += f"Mean: {mean_val:.3f}\n"
stat_string += f"Std Deviation: {std_val:.3f}\n"
stat_string += f"Min: {min_val:.3f}\n"
stat_string += f"Max: {max_val:.3f}\n"
if is_basis_finding and int(the_n) > 0:
    print("basis finding stats")
    group_key = vec_accept_count.index // (2 * int(the_n))
    block_maximums = vec_accept_count.groupby(group_key).max()
    mean_of_maxs = block_maximums.mean()
    stat_string += f"Mean of maximums per trial: {mean_of_maxs:.3f}\n"
for index, percentile_marker in enumerate(percentile_markers):
    stat_string += (
        f"{percentile_marker}th percentile: {percentiles[index]:.3f}\n"
    )
stat_string += f"Mean time per trial in ms: {durations.mean():.3f}\n"
print(stat_string)


outliers = df[df['VectorsAcceptedCount'] > percentiles[5]]
outliers = outliers.sort_values(by=['VectorsAcceptedCount'], ascending=True, inplace=False)
print(f"{outliers.shape[0]} outliers in unfiltered dataset (> 99th percentile)")
print(outliers)
if outliers.shape[0] > 0:
    basename_noext = os.path.splitext(os.path.basename(filename))[0]
    outliers.to_csv(f"{basename_noext}-outliers.csv", index=False)
    print(f"Wrote to {basename_noext}-outliers.csv")

if start_hist_from_min or use_log_scale:
    min_to_use = vec_accept_count.min()
else:
    min_to_use = 0

if hist_max and not use_log_scale:
    max_to_use = hist_max
else:
    max_to_use = vec_accept_count.max()

if use_log_scale:
    bins = np.logspace(np.log10(max(min_to_use, 0.01)), np.log10(max_to_use), num_bins)
else:
    bins = np.linspace(min_to_use, max_to_use, num_bins)

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(vec_accept_count, bins=bins, edgecolor='black')


if use_log_scale:
    plt.xscale('log')
    plt.xlabel('Number of vectors that u in U can be mapped to (log scale)')
    plt.xticks([1,2,4,8], ['1', '2', '4', '8'])
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    #ax.xaxis.set_major_locator(LogLocator(base=3.0, subs=(1.0,), numticks=10))
    #ax.xaxis.set_minor_locator(LogLocator(base=3.0, subs=(1.0,), numticks=10))
    #ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))

else:
    plt.xlabel('Number of vectors that u in U can be mapped to')
plt.ylabel('Frequency')
plt.minorticks_on()

plt.title(f'Histogram of necessary-condition accepted vectors on random bins (d={the_d}, n={the_n}, k={the_k}, bin size={the_vec_bin_size})')
plt.grid(True, linestyle='--', alpha=0.6)

image_name = f"{os.path.basename(filename)}.png"
plt.savefig(image_name, dpi=300)

stat_file_name = f"{os.path.basename(filename)}-stats.txt"
with open(stat_file_name, "w") as f:
    f.write(stat_string)

plt.close()
