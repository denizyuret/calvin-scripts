#!/usr/bin/env python
# Example:
# e = load_episodes()
# f = extract_features(e, range(0,3))
# f = extract_features(e, range(3,6), sincos=True) # default=False
# f = extract_features(e, range(3,6), stride=8)    # default=1
# rmsd1(f)

import numpy as np
import sys
from argparse import ArgumentParser

# Treatment of (1) distances, (2) angles, (3) binary features should be different!
# Squared error only makes sense if everybody is using the same units. Notes:
# - act[t]=tcp[t+1]
# - rel[t]=normalize_dist/angle(tcp[t+1]-tcp[t]))
# - normalize_dist=clip(act-obs,-0.02,0.02)/0.02
# - normalize_angle=clip(((act-obs) + pi) % (2*pi) - pi, -0.05, 0.05)/0.05

meter_features = ['actx','acty','actz','tcpx','tcpy','tcpz','tcpg','redx','redy','redz','bluex','bluey','bluez','pinkx','pinky','pinkz','slider.x','slider.y','slider.z','drawer.x','drawer.y','drawer.z','button.x','button.y','button.z','switch.x','switch.y','switch.z']
controller_features = ['slider','drawer','button','switch']  # these use relative (and for button, scaled) coordinates, see calvin_controller_xyz.py
radian_features = ['acta','actb','actc','tcpa','tcpb','tcpc','arm1','arm2','arm3','arm4','arm5','arm6','arm7','reda','redb','redc','bluea','blueb','bluec','pinka','pinkb','pinkc']
binary11_features = ['actg','relg','armg'] # (-1/1); actg and relg identical, refer to action, armg refers to sensing
binary01_features = ['lightbulb','greenlight']  # (0/1) binary
rel_meter_features = ['relx','rely','relz']     # unit=2cm, cropped to +-1
rel_angle_features = ['rela','relb','relc']     # unit=0.05 radians, cropped to +-1
depth_tactile_features = ['tact1d', 'tact2d']   # depth normalized with x100
rgb_tactile_features = ['tact1r', 'tact1g', 'tact1b', 'tact2r', 'tact2g', 'tact2b'] # rgb normalized with /255.0


def rmsd_all(path='data/D-training.npz', stride=1, degree=1):
    d = np.load(path, allow_pickle=True)
    fieldnames = d['fieldnames']
    episodes = load_episodes(path)
    rmsd_fns = [ rmsd0, rmsd1, rmsd2 ]
    rmsd_fn = rmsd_fns[degree]
    print("colnum\tfeature\trmsd1\trmsd1angle")
    for i in range(0,97):
        err_dist = rmsd_fn(extract_features(episodes, range(i,i+1), sincos=False, stride=stride))
        err_angle = rmsd_fn(extract_features(episodes, range(i,i+1), sincos=True, stride=stride))
        print(f"{i}\t{fieldnames[i]}\t{err_dist:.4f}\t{err_angle:.4f}")
        

def load_episodes(path='data/D-training.npz'):
    d = np.load(path)
    episode_start_end = sorted(d['episodes'].tolist())
    f2i = np.full(1+max(d['frameids']), -1)
    for i,f in enumerate(d['frameids']):
        f2i[f] = i
    episodes = []
    data = d['data']
    frameids = d['frameids']
    for f_start,f_end in episode_start_end:
        # make sure we have all the frames
        i_start, i_end = f2i[f_start], f2i[f_end]
        fids = frameids[i_start:i_end]
        assert np.array_equal(fids[1:], 1+fids[:-1])
        episodes.append(data[i_start:i_end,:])
    return episodes


def extract_features(episodes, features=range(0,97), sincos=False, stride=1):
    data = []
    for d in episodes:
        d = d[:, features]
        if stride > 1:
            d = d[::stride,:]
        if sincos:
            d = np.concatenate((np.sin(d), np.cos(d)), axis=1)
        data.append(d)
    return data
        

def rmsd0(data):
    """compute rmsd for constant position model that assumes f[t+1] = f[t]"""
    sqerr, cnt = 0, 0
    for d in data:
        targets = d[1:,:]
        preds = d[:-1,:]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return np.sqrt(sqerr/cnt)

def rmsd1(data):                # gives the best results
    """compute rmsd for constant speed model that assumes f[t+1]-f[t]==f[t]-f[t-1]"""
    # x2-x1 = x1-x0 => x2 = 2x1-x0
    sqerr, cnt = 0, 0
    for d in data:          # (T,F)
        targets = d[2:,:]
        preds = 2*d[1:-1,:] - d[:-2,:]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return np.sqrt(sqerr/cnt)

def rmsd2(data):
    """compute rmsd for constant acceleration model that assumes (f[t+1]-f[t]) - (f[t]-f[t-1]) = (f[t]-f[t-1]) - (f[t-1]-f[t-2])"""
    # (x3-x2) - (x2-x1) = (x2-x1) - (x1-x0)
    # (x3-x2) = 2(x2-x1) - (x1-x0)
    # x3 = 3x2 - 3x1 + x0
    sqerr, cnt = 0, 0
    for d in data:          # (T,F)
        targets = d[3:,:]
        preds = 3*d[2:-1,:] - 3*d[1:-2,:] + d[:-3,:]
        sqerr += np.sum(np.square(targets-preds))
        cnt += targets.size
    return np.sqrt(sqerr/cnt)

if __name__ == "__main__":
    parser = ArgumentParser(description='Compute ssl baselines')
    parser.add_argument('-d', '--data', type=str, default='data/D-validation.npz', help='Path to npz file with data.')
    # parser.add_argument('-a', '--angle', action='store_true', help='Transform each feature x to sin(x),cos(x)')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Take every stride timesteps from data')
    # parser.add_argument('-f', '--features', type=str, default='range(0,97)', help='Features to extract')
    parser.add_argument('-n', '--degree', type=int, default=1, help='Model degree to compute rmsd for, 0,1,2 supported.')
    args = parser.parse_args()
    # args.features = eval(args.features)
    rmsd_all(args.data, stride=args.stride, degree=args.degree)
