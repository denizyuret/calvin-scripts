# cd /datasets/calvin/debug/train
# python calvin_extract_numbers.py > debug-train.tsv

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
        actions = data.get("actions")  # 1-7(7)
        rel_actions = data.get("rel_actions")  # 8-14(7)
        robot_obs = data.get("robot_obs")  # 15-29(15)
        scene_obs = data.get("scene_obs")  # 30-53(24)
        print(idnum, end='\t')
        print(*np.concatenate((actions, rel_actions, robot_obs, scene_obs)), sep='\t')
