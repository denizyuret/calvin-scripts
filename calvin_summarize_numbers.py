# usage: python calvin_summarize_numbers.py data.npz
# prints statistics for each column

import pandas as pd
import numpy as np
import sys

d = np.load(sys.argv[1], allow_pickle=True)
df = pd.DataFrame(d['data'], columns=d['fieldnames'])
stats = df.describe() # .transpose()
print(stats.to_string())
