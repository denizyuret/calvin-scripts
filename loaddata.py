import sys
import numpy as np
import torch
import gzip
from torch.utils.data import TensorDataset, Dataset


class CalvinDataset(Dataset):
    """
    Turn the last `instances_per_episode` frames of each episode into instances with `features` from the last `context_length` frames.
    Version 3: This has the same behavior as calvindataset() but is more memory efficient.
    It keeps only the raw data in memory and constructs the context for each instance as needed.
    """
    def __init__(self, prefix='data/debug-training', features=range(1,98), instances_per_episode=1, context_length=64):
        # There are three indices for each frame:
        # 1. Frame id given in the filename (and first column of data), which may be discontinuous and not start from 0.
        # 2. Position in the `data` array. pos2id[2]=1 & id2pos[1]=2.
        # 3. Position in the final dataset, which contains only the annotated subset of data. data_index[3]=2
        data,pos2id,id2pos = loaddata(prefix)
        lang,task2int,int2task = loadlang(prefix)
        data_index = []; target = []; frame = []
        for (start_frame, end_frame, task, annot) in lang:
            taskid = task2int[task]
            for instance_frame in range(end_frame - instances_per_episode + 1, end_frame+1):
                instance_index = id2pos[instance_frame]
                if instance_index - context_length + 1 >= 0:
                    frame.append(instance_frame)
                    target.append(taskid)
                    data_index.append(instance_index)
        self.__dict__.update(locals()) # convert each local variable to self.var

    def __getitem__(self, index):
        i = self.data_index[index]
        x = self.data[np.ix_(range(1+i-self.context_length, 1+i), self.features)]
        inputs = torch.from_numpy(x)
        target = torch.tensor(self.target[index])
        frame = torch.tensor(self.frame[index])
        return (inputs, target, frame)

    def __len__(self):
        return len(self.target)


def calvindataset(prefix='data/debug-training', features=range(1,98), instances_per_episode=32, context_length=1):
    """
    Turn the last `instances_per_episode` frames of each episode into instances with `features` from the last `context_length` frames.
    Version 2: generalizes the three functions calvindataset1,2,3.
    To get equivalent output to the previous version:
    calvindataset1(..., window=32) = calvindataset(..., instances_per_episode=32, context_length=1)
    calvindataset2(..., window=32) = calvindataset(..., instances_per_episode=1, context_length=32)
    ... and except for reordering of features:
    calvindataset3(..., window=32, frames=4) = calvindataset(..., instances_per_episode=32, context_length=4)
    """
    data,pos2id,id2pos = loaddata(prefix)
    lang,task2int,int2task = loadlang(prefix)
    x = []; y = []; idx = []
    for (i,j,task,annot) in lang: # episode starts at i, ends at j, classified as task.
        taskid = task2int[task]
        for k in range(j-instances_per_episode+1, j+1): # we create instances for each of the last n frames of an episode
            p = id2pos[k] - context_length + 1
            if p >= 0:
                idx.append(k)
                y.append(taskid)
                x.append(data[np.ix_(range(p, p+context_length), features)].reshape(-1))
    x = torch.tensor(np.stack(x))
    y = torch.tensor(y)
    idx = torch.tensor(idx)
    return TensorDataset(x,y,idx)


def calvindataset1(prefix='data/debug-training', features=range(1,98), window=32):
    """
    Predict task from a single frame (picked anywhere from the last 32 frames associated with task)
    Version 1: deprecated, please use CalvinDataset(features, instances_per_episode=32, context_length=1)
    """
    data,pos2id,id2pos = loaddata(prefix)
    lang,task2int,int2task = loadlang(prefix)
    p = []
    y = []
    idx = []
    for (i,j,task,annot) in lang: # i,j,k frame ids. p is an index into data.
        taskid = task2int[task]
        for k in range(j-window+1,j+1):
            p.append(id2pos[k])
            y.append(taskid)
            idx.append(k)
    x = torch.tensor(data[np.ix_(p,features)])
    y = torch.tensor(y)
    idx = torch.tensor(idx)
    return TensorDataset(x,y,idx)



def calvindataset2(prefix='data/debug-training', features=range(1,98), window=32):
    """
    Predict annotation from the last 32 frames (out of 64 associated with the annotation).
    Version 1: deprecated, please use CalvinDataset(features, instances_per_episode=1, context_length=32)
    """
    data,pos2id,id2pos = loaddata(prefix)
    lang,task2int,int2task = loadlang(prefix)
    x = []
    y = []
    idx = []
    for (i,j,task,annot) in lang:
        taskid = task2int[task]
        p = id2pos[j]
        x.append(np.ravel(data[np.ix_(range(p-window+1,p+1), features)]))
        y.append(taskid)
        idx.append(j)
    x = torch.tensor(np.stack(x))
    y = torch.tensor(y)
    idx = torch.tensor(idx)
    return TensorDataset(x,y,idx)



def calvindataset3(prefix='data/debug-training', features=range(1,98), window=32, frames=4):
    """
    Predict annotation from 4 successive frames (picked anywhere from the last 32 frames associated with task).
    Version 1: deprecated, please use CalvinDataset(features, instances_per_episode=32, context_length=4)
    """
    data,pos2id,id2pos = loaddata(prefix)
    lang,task2int,int2task = loadlang(prefix)
    p = []
    y = []
    idx = []
    for (i,j,task,annot) in lang:
        taskid = task2int[task]
        for k in range(j-window+1,j+1):
            p.append(id2pos[k])
            y.append(taskid)
            idx.append(k)
    p = np.array(p)
    x = []
    for f in range(0, frames):
        x.append(torch.tensor(data[np.ix_(p-f, features)]))
    x = torch.cat(x, dim=1)
    y = torch.tensor(y)
    idx = torch.tensor(idx)
    return TensorDataset(x,y,idx)


def loaddata(prefix):
    print(f"Loading {prefix}", file=sys.stderr)
    with gzip.open(prefix + '.tsv.gz', 'rt') as f:
        data = np.loadtxt(f, delimiter='\t', dtype='float32') # not using dtype=dtype_data: it makes a 1-D record array instead of 2-D
    with gzip.open(prefix + '-controllers.tsv.gz', 'rt') as f:
        cont = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], cont[:,0]), 'cont indices do not match'
    with gzip.open(prefix + '-tactile2.tsv.gz', 'rt') as f:
        tact = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], tact[:,0]), 'tact indices do not match'
    with gzip.open(prefix + '-sdiff.tsv.gz', 'rt') as f:
        sdiff = np.loadtxt(f, delimiter='\t', dtype='float32')
        assert np.array_equal(data[:,0], sdiff[:,0]), 'sdiff indices do not match'
    data = np.concatenate((data, cont[:,1:], tact[:,1:], sdiff[:,1:]), axis=1)
    data = normalize(data)
    pos2id = data[:,0].astype(int)
    id2pos = np.full(1+max(pos2id), -1)
    for (pos, id) in enumerate(pos2id):
        id2pos[id] = pos
    return data, pos2id, id2pos


def normalize(data):
    data[:,32:34] = data[:,32:34] * 10.0  # button and switch
    data[:,21] = data[:,21] * 10.0        # gripper opening
    return data


def loadlang(prefix):
    print(f"Loading {prefix}-lang", file=sys.stderr)
    with gzip.open(prefix + '-lang.tsv.gz', 'rt') as f:
        lang = np.loadtxt(f, delimiter='\t', dtype=dtype_lang)
    lang.sort(order = 'end')
    task2int = {}
    for (i,task) in enumerate(int2task):
        task2int[task] = i
    for task in lang['task']:
        assert task in task2int, 'task not found'
    return lang, task2int, int2task

dtype_lang = [
    ('start', int),
    ('end', int),
    ('task', object),
    ('annot', object)
]


dtype_data = [
    ('idnum', int),  # 00. 
    ('actx', float), # 01.  (tcp (tool center point) position (3): x,y,z in absolute world coordinates)
    ('acty', float), # 02. 
    ('actz', float), # 03. 
    ('acta', float), # 04.  (tcp orientation (3): euler angles a,b,c in absolute world coordinates)
    ('actb', float), # 05. 
    ('actc', float), # 06. 
    ('actg', int),   # 07.  (gripper_action (1): binary close=-1, open=1)
    ('relx', float), # 08.  (tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50)
    ('rely', float), # 09. 
    ('relz', float), # 10. 
    ('rela', float), # 11.  (tcp orientation (3): euler angles a,b,c in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20)
    ('relb', float), # 12. 
    ('relc', float), # 13. 
    ('relg', int),   # 14.  (gripper_action (1): binary close=-1, open=1)
    ('tcpx', float), # 15.  (tcp position (3): x,y,z in world coordinates)
    ('tcpy', float), # 16. 
    ('tcpz', float), # 17. 
    ('tcpa', float), # 18.  (tcp orientation (3): euler angles a,b,c in world coordinates)
    ('tcpb', float), # 19. 
    ('tcpc', float), # 20. 
    ('tcpg', float), # 21.  (gripper opening width (1): in meters)
    ('arm1', float), # 22.  (arm_joint_states (7): in rad)
    ('arm2', float), # 23. 
    ('arm3', float), # 24. 
    ('arm4', float), # 25. 
    ('arm5', float), # 26. 
    ('arm6', float), # 27. 
    ('arm7', float), # 28. 
    ('armg', int),   # 29.  (gripper_action (1): binary close = -1, open = 1)
    ('slider', float), # 30.  (1): joint state: range=[-0.002359:0.306696]
    ('drawer', float), # 31.  (1): joint state: range=[-0.002028:0.221432]
    ('button', float), # 32.  (1): joint state: range=[-0.000935:0.033721]
    ('switch', float), # 33.  (1): joint state: range=[-0.004783:0.091777]
    ('lightbulb', int),  # 34.  (1): on=1, off=0
    ('greenlight', int), # 35.  (1): on=1, off=0
    ('redx', float), # 36.  (red block (6): (x, y, z, euler_x, euler_y, euler_z)
    ('redy', float), # 37. 
    ('redz', float), # 38. 
    ('reda', float), # 39. 
    ('redb', float), # 40. 
    ('redc', float), # 41. 
    ('bluex', float), # 42.  (blue block (6): (x, y, z, euler_x, euler_y, euler_z)
    ('bluey', float), # 43. 
    ('bluez', float), # 44. 
    ('bluea', float), # 45. 
    ('blueb', float), # 46. 
    ('bluec', float), # 47. 
    ('pinkx', float), # 48.  (pink block (6): (x, y, z, euler_x, euler_y, euler_z)
    ('pinky', float), # 49. 
    ('pinkz', float), # 50. 
    ('pinka', float), # 51. 
    ('pinkb', float), # 52. 
    ('pinkc', float)  # 53. 
]

dtype_cont = [
    ('idnum', int),      # 00.00
    ('slider.x', float), # 01.54
    ('slider.y', float), # 02.55
    ('slider.z', float), # 03.56
    ('drawer.x', float), # 04.57
    ('drawer.y', float), # 05.58
    ('drawer.z', float), # 06.59
    ('button.x', float), # 07.60
    ('button.y', float), # 08.61
    ('button.z', float), # 09.62
    ('switch.x', float), # 10.63
    ('switch.y', float), # 11.64
    ('switch.z', float)  # 12.65
]

dtype_tact = [
    ('idnum', int),    # 00.00 idnum
    ('tact1d', float), # 01.66 depth_tactile1
    ('tact2d', float), # 02.67 depth_tactile2
    ('tact1r', float), # 03.68 rgb_tactile1_r
    ('tact1g', float), # 04.69 rgb_tactile1_g
    ('tact1b', float), # 05.70 rgb_tactile1_b
    ('tact2r', float), # 06.71 rgb_tactile2_r
    ('tact2g', float), # 07.72 rgb_tactile2_g
    ('tact2b', float)  # 08.73 rgb_tactile2_b
]

dtype_sdiff = [
    ('idnum', int),      # 00.00
    ('slider_diff', float),   # 01.74
    ('drawer_diff', float),   # 02.75
    ('button_diff', float),   # 03.76
    ('switch_diff', float),   # 04.77
    ('lightbulb_diff', int),  # 05.78
    ('greenlight_diff', int), # 06.79
    ('redx_diff', float),     # 07.80
    ('redy_diff', float),     # 08.81
    ('redz_diff', float),     # 09.82
    ('reda_diff', float),     # 10.83
    ('redb_diff', float),     # 11.84
    ('redc_diff', float),     # 12.85
    ('bluex_diff', float),    # 13.86
    ('bluey_diff', float),    # 14.87
    ('bluez_diff', float),    # 15.88
    ('bluea_diff', float),    # 16.89
    ('blueb_diff', float),    # 17.90
    ('bluec_diff', float),    # 18.91
    ('pinkx_diff', float),    # 19.92
    ('pinky_diff', float),    # 20.93
    ('pinkz_diff', float),    # 21.94
    ('pinka_diff', float),    # 22.95
    ('pinkb_diff', float),    # 23.96
    ('pinkc_diff', float)     # 24.97
]    

act_range = range(1,8)
rel_range = range(8,15)
tcp_range = range(15,22)
arm_range = range(22,30)
scene_range = range(30,54)
controller_range = range(54,66)
tactile_range = range(66,74)
sdiff_range = range(74,98)

int2task = [
    'close_drawer',
    'lift_blue_block_drawer',
    'lift_blue_block_slider',
    'lift_blue_block_table',
    'lift_pink_block_drawer',
    'lift_pink_block_slider',
    'lift_pink_block_table',
    'lift_red_block_drawer',
    'lift_red_block_slider',
    'lift_red_block_table',
    'move_slider_left',
    'move_slider_right',
    'open_drawer',
    'place_in_drawer',
    'place_in_slider',
    'push_blue_block_left',
    'push_blue_block_right',
    'push_into_drawer',
    'push_pink_block_left',
    'push_pink_block_right',
    'push_red_block_left',
    'push_red_block_right',
    'rotate_blue_block_left',
    'rotate_blue_block_right',
    'rotate_pink_block_left',
    'rotate_pink_block_right',
    'rotate_red_block_left',
    'rotate_red_block_right',
    'stack_block',
    'turn_off_led',
    'turn_off_lightbulb',
    'turn_on_led',
    'turn_on_lightbulb',
    'unstack_block'
]

