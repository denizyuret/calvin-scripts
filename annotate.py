# Usage: zcat preds.tsv.gz | python annotate.py -t 0.95 -l 20 | gzip > annotations.tsv.gz
# Generate an annotation file from model probabilities
# According to probability.py, a good rule of thumb is to annotate intervals where one task has >95% probability for at least 20 frames.

import sys
import gzip
import numpy as np
import loaddata as ld
from argparse import ArgumentParser

parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
parser.add_argument("-t", "--threshold", default=0.95, type=float, help="Minimum probability to start an episode.")
parser.add_argument("-l", "--length", default=20, type=int, help="Minimum length of interval above threshold.")
args = parser.parse_args()

preds = np.loadtxt(sys.stdin, delimiter='\t', dtype='float32')
frames = preds[:,0].astype(np.int32)
probs = preds[:,1:]

taskid = -1
taskcnt = 0

for i in range(len(frames)):
    sorted_indices = np.argsort(probs[i,:])
    if probs[i, sorted_indices[-1]] >= args.threshold:
        if taskid != sorted_indices[-1]:
            taskid = sorted_indices[-1]
            taskcnt = 1
        else:
            taskcnt = taskcnt + 1
    else:
        if taskcnt >= args.length and i >= 64:
            print(f"{frames[i-64]}\t{frames[i-1]}\t{ld.int2task[taskid]}\tauto")
        taskcnt = 0
        taskid = -1
