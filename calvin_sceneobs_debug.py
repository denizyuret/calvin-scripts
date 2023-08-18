import numpy as np

trn = np.load("data/D-training.npz", allow_pickle=True)
val = np.load("data/D-validation.npz", allow_pickle=True)
task = "rotate_red_block_right"
taskid = np.where(trn["tasknames"] == task)[0][0] # "rotate_red_block_right" = 27
trnidx = { frame: index for index, frame in enumerate(trn["frameids"]) }
validx = { frame: index for index, frame in enumerate(val["frameids"]) }
trn_start_frameids = trn["lang"][trn["lang"]["taskid"] == taskid]["end"] - 31
trn_start_indices = [ trnidx[frameid] for frameid in trn_start_frameids ]
val_start_frameids = val["lang"][val["lang"]["taskid"] == taskid]["end"] - 31
val_start_indices = [ validx[frameid] for frameid in val_start_frameids ]
test_start_frameids = [8338, 8340, 13006, 16408, 16419, 16818, 36382, 71207, 71217, 82352]
test_val_start_indices = [ validx[frameid] for frameid in test_start_frameids if frameid in validx ]
test_trn_start_indices = [ trnidx[frameid] for frameid in test_start_frameids if frameid in trnidx ]
fld = { string: index for index, string in enumerate(trn["fieldnames"]) }
flds = [ fld[name] for name in ["tcpx", "tcpy", "tcpz", "redx", "redy", "redz"] ]
trn_flds = trn["data"][trn_start_indices][:, flds]
val_flds = val["data"][val_start_indices][:, flds]
test_val_flds = val["data"][test_val_start_indices][:, flds]
test_trn_flds = trn["data"][test_trn_start_indices][:, flds]
test_flds = np.vstack((test_val_flds, test_trn_flds))

