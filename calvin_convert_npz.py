#!/usr/bin/env python
# python calvin_convert_npz.py D validation
# Convert tsv data to npz format

import sys
import numpy as np
import loaddata as ld
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert CALVIN dataset to npz format")
parser.add_argument("dataset", type=str, default="debug", help="debug|D|ABC|ABCD")
parser.add_argument("split", type=str, default="validation", help="training|validation")
parser.add_argument("-c", "--calvin", type=str, default="/datasets/calvin", help="Path to original CALVIN data root")
parser.add_argument("-d", "--data", type=str, default="data", help="Directory of extracted .tsv.gz files")
args = parser.parse_args()

ep_start_end_path = f"{args.calvin}/{args.dataset}/{args.split}/ep_start_end_ids.npy"
print(f"Reading {ep_start_end_path}...", file=sys.stderr)
ep_start_end_ids = np.load(ep_start_end_path)

tsv_gz_path = f"{args.data}/{args.dataset}-{args.split}"
print(f"Reading {tsv_gz_path}...", file=sys.stderr)
data, pos2id, id2pos = ld.loaddata(tsv_gz_path)
lang, task2int, int2task = ld.loadlang(tsv_gz_path)
assert np.array_equal(pos2id, data[:,0].astype(int))

print(f"Cooking data...", file=sys.stderr)
# replace task names with task ids in lang:
lang = np.array([(start, end, task2int[task], annot) for (start, end, task, annot) in lang],
                dtype=[('start', '<i4'), ('end', '<i4'), ('taskid', '<i8'), ('instruction', 'O')])

# cut the buggy extra frames from ABCD-training:
if args.dataset == "ABCD" and args.split == "training":
    searchlist = pos2id.tolist()
    delete_rows = [ searchlist.index(x) for x in (37682, 53818, 244284, 420498) ]
    data = np.delete(data, delete_rows, axis=0)

# first column of data is frame_id and should be cut
frameids = data[:,0].astype('int32')
data = data[:,1:]

# collect column names and task names
fieldnames = np.array([x[0] for x in ld.dtype_data[1:] + ld.dtype_cont[1:] + ld.dtype_tact[1:] + ld.dtype_sdiff[1:]], dtype='O')
tasknames = np.array(int2task, dtype='O')
episodes = ep_start_end_ids.astype('int32')

output_path = f"{args.data}/{args.dataset}-{args.split}.npz"
print(f"Saving to {output_path}...", file=sys.stderr)
np.savez_compressed(output_path, data=data, lang=lang,
                    frameids=frameids,
                    tasknames=tasknames,
                    fieldnames=fieldnames,
                    episodes=episodes)
