# python calvin_scene_check.py ABCD/training
# Find instances of red/pink/blue mixup comparing scene coordinates and tcp coordinates

import re
import sys
import gzip
import torch
import numpy as np
import loaddata as ld
from torch.linalg import norm

#ep_start_end_path = '/datasets/calvin/' + sys.argv[1] + '/ep_start_end_ids.npy'
#ep_start_end = np.load(ep_start_end_path)

tsv_gz_path = re.sub(r"(.+?)/(.+)", r"data/\1-\2", sys.argv[1])
dataset = ld.CalvinDataset(tsv_gz_path, context_length=32, features=range(0,53))

colors = ['red', 'blue', 'pink']
classes = [[i for (i,x) in enumerate(ld.int2task) if colors[j] in x] for j in range(len(colors))]
coordinates = [range(36,39), range(42,45), range(48,51)]
tcp_coordinates = range(15,18)

for (x,y,i) in dataset:
    dist = [ norm(x[:,tcp_coordinates] - x[:,coordinates[c]]) for c in range(len(colors)) ]
    mindist_color = dist.index(min(dist))
    stdev = [ torch.std(x[:,coordinates[c][0]]) +
              torch.std(x[:,coordinates[c][1]]) +
              torch.std(x[:,coordinates[c][2]])
              for c in range(len(colors)) ]
    maxstd_color = stdev.index(max(stdev))
    for c in range(len(colors)):
        if y in classes[c] and (c != mindist_color or c != maxstd_color):
            print(f"{i}\t{ld.int2task[y]}\tmindist={colors[mindist_color]}\tmaxstd={colors[maxstd_color]}")

    
# with gzip.open(tsv_gz_path, 'rt') as f:
#     data = np.loadtxt(f, delimiter='\t', dtype='float32')

# ep_end = {}
# for i in range(ep_start_end.shape[0]):
#     ep_end[ep_start_end[i,1]] = True

# def normalize(diff):
#     """based on https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/utils/utils.py#L160"""
#     meter_fields = [0,1,2,3,6,7,8,12,13,14,18,19,20]
#     radian_fields = [9,10,11,15,16,17,21,22,23]
#     boolean_fields = [4,5]
#     for i in meter_fields:
#         diff[i] = np.clip(diff[i], -0.02, 0.02) / 0.02 # 0.02m = 2cm = 1
#         if abs(diff[i]) < 1e-3: # clean up noise < 0.02 mm
#             diff[i] = 0
#     for i in radian_fields:
#         diff[i] = np.clip((diff[i] + np.pi) % (2*np.pi) - np.pi, -0.05, 0.05) / 0.05 # 0.05 rad = 2.86 deg = 1
#         if abs(diff[i]) < 1e-2: # clean up noise < 0.02 degrees
#             diff[i] = 0
#     return diff


# for i in range(data.shape[0]):
#     curr_frame = int(data[i,0])
#     curr_scene = data[i,30:54]
#     if curr_frame in ep_end:
#         diff = np.zeros(len(curr_scene), dtype='float32')
#     else:
#         next_scene = data[i+1,30:54]
#         diff = normalize(next_scene - curr_scene)

#     print(f"{curr_frame:07d}", end="")
#     if True: # for dataset
#         for d in diff:
#             print(f"\t{d:g}", end="")
#     else:    # for visualization
#         print("", end="\t")
#         for d in diff:
#             print(f" {d:g}", end="")
#     print("")
