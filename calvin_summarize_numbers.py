# usage: python calvin_summarize_numbers.py < data.tsv
# prints statistics for each column

import pandas as pd
import sys

df = pd.read_csv(sys.stdin, sep='\t', header=None)
stats = df.describe().transpose()
print(stats.to_string())
