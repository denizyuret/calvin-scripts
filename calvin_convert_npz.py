#!/usr/bin/env python
# python calvin_convert_npz.py D validation
# Convert tsv data to npz format

import sys
import gzip
import numpy as np
import loaddata as ld
from argparse import ArgumentParser


def loaddata_simple(prefix):    # load without normalization
    with gzip.open(prefix + '.tsv.gz', 'rt') as f:
        data = np.loadtxt(f, delimiter='\t', dtype='float32')
    with gzip.open(prefix + '-controllers.tsv.gz', 'rt') as f:
        cont = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], cont[:,0]), 'cont indices do not match'
    with gzip.open(prefix + '-tactile2.tsv.gz', 'rt') as f:
        tact = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], tact[:,0]), 'tact indices do not match'
    with gzip.open(prefix + '-sdiff.tsv.gz', 'rt') as f:
        sdiff = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], sdiff[:,0]), 'sdiff indices do not match'
    return np.concatenate((data, cont[:,1:], tact[:,1:], sdiff[:,1:]), axis=1)


parser = ArgumentParser(description="Convert CALVIN dataset to npz format")
parser.add_argument("dataset", type=str, default="debug", help="debug|D|ABC|ABCD")
parser.add_argument("split", type=str, default="validation", help="training|validation")
parser.add_argument("-c", "--calvin", type=str, default="/datasets/calvin", help="Path to original CALVIN data root")
parser.add_argument("-d", "--data", type=str, default="data", help="Directory of extracted .tsv.gz files")
args = parser.parse_args()

ep_start_end_path = f"{args.calvin}/{args.dataset}/{args.split}/ep_start_end_ids.npy"
print(f"Reading {ep_start_end_path}...", file=sys.stderr)
episodes = np.load(ep_start_end_path).astype('int32')

tasknames = np.array(ld.int2task, dtype='O')
task2int = {}
for (i,task) in enumerate(ld.int2task):
    task2int[task] = i

lang_path = f"{args.calvin}/{args.dataset}/{args.split}/lang_annotations/auto_lang_ann.npy"
print(f"Reading {lang_path}...", file=sys.stderr)
a = np.load(lang_path, allow_pickle=True).item()
lang = np.array([(start, end, task2int[task], annot) for
                 ((start, end), task, annot) in zip(a['info']['indx'], a['language']['task'], a['language']['ann'])],
                dtype=[('start', '<i4'), ('end', '<i4'), ('taskid', '<i8'), ('instruction', 'O')])
lang.sort(order='end')
         
tsv_gz_path = f"{args.data}/{args.dataset}-{args.split}"
print(f"Reading {tsv_gz_path}...", file=sys.stderr)
data = loaddata_simple(tsv_gz_path)
fieldnames = np.array([x[0] for x in ld.dtype_data[1:] + ld.dtype_cont[1:] + ld.dtype_tact[1:] + ld.dtype_sdiff[1:]], dtype='O')

# cut the buggy extra frames from ABCD-training:
if args.dataset == "ABCD" and args.split == "training":
    searchlist = data[:,0].astype('int32').tolist()
    delete_rows = [ searchlist.index(x) for x in (37682, 53818, 244284, 420498) ]
    data = np.delete(data, delete_rows, axis=0)

# first column of data is frame_id and should be cut
frameids = data[:,0].astype('int32')
data = data[:,1:]

output_path = f"{args.data}/{args.dataset}-{args.split}.npz"
print(f"Saving to {output_path}...", file=sys.stderr)
np.savez_compressed(output_path, data=data, lang=lang,
                    frameids=frameids,
                    tasknames=tasknames,
                    fieldnames=fieldnames,
                    episodes=episodes)
