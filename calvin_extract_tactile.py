# cd /datasets/calvin/debug/train
# python calvin_extract_tactile.py > debug-train.tsv

import numpy as np
import os
import re
import sys
from tqdm import tqdm

for f in tqdm(sorted(os.listdir('.'))):
    m = re.match(r"episode_(\d{7})\.npz", f)
    if m is not None:
        idnum = m.group(1)
        data = np.load(f, allow_pickle=True, mmap_mode='r')
        depth_tactile = data.get('depth_tactile')
        rgb_tactile = data.get('rgb_tactile')
        print(idnum, end='')
        for i in range(2):
            r = depth_tactile[:,:,i].mean() * 100.0
            print(f"\t{r:g}", end='')
        for i in range(6):
            r = rgb_tactile[:,:,i].mean() / 255.0
            print(f"\t{r:g}", end='')
        print('')
