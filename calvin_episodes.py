# zcat D-validation.tsv.gz | python calvin_episodes.py
# Tries to guess the episode boundaries if xyz of successive frames differ by more than 8.5 std.

import numpy as np
import sys

# Standard deviations computed on D/validation
# The 8.5 limit does not work perfectly (misses 1 out of 31 episode boundaries in D/training).
stdevs = np.array([0.005631, 0.004607, 0.004377, 1.053450, 0.008449, 0.102429, 0.315108, 0.065267, 0.052577, 0.062196, 0.122271, 0.086286, 0.093735, 0.315108, 0.005632, 0.004607, 0.004377, 1.053827, 0.008445, 0.102424, 0.006102, 0.016108, 0.008584, 0.014204, 0.016424, 0.010331, 0.008892, 0.013924, 0.315108, 0.002641, 0.001932, 0.000573, 0.000869, 0.034667, 0.038530, 0.001958, 0.002136, 0.001393, 0.871913, 0.009788, 0.267081, 0.002307, 0.002118, 0.001470, 0.917085, 0.011361, 0.520260, 0.002117, 0.002326, 0.001320, 0.784371, 0.008722, 0.081872])
interval_start = -1
last_index = -1
last_value = -1

for line in sys.stdin:
    fields = line.strip().split('\t')
    index = int(fields[0])
    value = np.array([ float(val) for val in fields[1:] ])
    if interval_start == -1:
        interval_start = index
    elif index > last_index + 1 or np.abs((value[0:3] - last_value[0:3]) / stdevs[0:3]).max() > 8.5:
        print(interval_start, '\t', last_index)
        interval_start = index
    last_index = index
    last_value = value

print(interval_start, '\t', last_index)

