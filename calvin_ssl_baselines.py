#!/usr/bin/env python
# Example:
# e = load_episodes()
# rmsd0(e, range(0,97))
# TODO: treatment of (1) distances, (2) angles, (3) binary features should be different!

import math
import numpy as np
import torch
import calvin_dataset as cd

def load_episodes(path='data/D-training.npz'):
    d = np.load(path)
    episode_start_end = sorted(d['episodes'].tolist())
    f2i = np.full(1+max(d['frameids']), -1)
    for i,f in enumerate(d['frameids']):
        f2i[f] = i
    episodes = []
    for f_start,f_end in episode_start_end:
        # make sure we have all the frames
        i_start, i_end = f2i[f_start], f2i[f_end]
        fids = d['frameids'][i_start:i_end]
        assert np.array_equal(fids[1:], 1+fids[:-1])
        episodes.append(d['data'][i_start:i_end,:])
    return episodes

def rmsd0(episodes, features):
    """compute rmsd for constant position model that assumes f[t+1] = f[t]"""
    sqerr, cnt = 0, 0
    for d in episodes:          # (T,F)
        targets = d[1:,features]
        preds = d[:-1,features]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return math.sqrt(sqerr/cnt)

def rmsd1(episodes, features):
    """compute rmsd for constant speed model that assumes f[t+1]-f[t]==f[t]-f[t-1]"""
    # x2-x1 = x1-x0 => x2 = 2x1-x0
    sqerr, cnt = 0, 0
    for d in episodes:          # (T,F)
        targets = d[2:,features]
        preds = 2*d[1:-1,features] - d[:-2,features]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return math.sqrt(sqerr/cnt)

def rmsd2(episodes, features):
    """compute rmsd for constant acceleration model that assumes (f[t+1]-f[t]) - (f[t]-f[t-1]) = (f[t]-f[t-1]) - (f[t-1]-f[t-2])"""
    # (x3-x2) - (x2-x1) = (x2-x1) - (x1-x0)
    # (x3-x2) = 2(x2-x1) - (x1-x0)
    # x3 = 3x2 - 3x1 + x0
    sqerr, cnt = 0, 0
    for d in episodes:          # (T,F)
        targets = d[3:,features]
        preds = 3*d[2:-1,features] - 3*d[1:-2,features] + d[:-3,features]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return math.sqrt(sqerr/cnt)
