# usage: python calvin_summarize_numbers.py data.tsv

import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], sep='\t', header=None)
stats = df.describe().transpose()
print(stats.to_string())
