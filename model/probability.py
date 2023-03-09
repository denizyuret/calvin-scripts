# Usage: python probability.py
# Statistics on how model probability for the correct class evolves [-64:20] frames around the end of labeled episode

import sys
import gzip
import numpy as np
import pandas as pd
import loaddata as ld

lang, task2int, int2task = ld.loadlang('../data/ABCD-validation')

print('Loading ../data/ABCD-validation-preds.tsv.gz', file=sys.stderr)

with gzip.open('../data/ABCD-validation-preds.tsv.gz', 'rt') as f:
    preds = np.loadtxt(f, delimiter='\t', dtype='float32')

preds_index = 0
probs = []

for (start, end, task, annot) in lang:
    taskid = task2int[task]
    while(preds[preds_index,0] < end):
        preds_index += 1
    if preds_index < 63 or int(preds[preds_index,0]) != end:
        continue
    probs.append(preds[preds_index-63:preds_index+20, taskid+1])

print(pd.DataFrame(np.stack(probs)).describe().transpose().to_string())

