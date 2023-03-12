import numpy as np
import torch
from torch.utils.data import Dataset


class CalvinDataset(Dataset):
    """
    Turn the last `instances_per_episode` frames of each episode into instances 
    with a subset of `features` from the last `context_length` frames.
    """
    def __init__(self, path='data/debug-training.npz', features=range(0,97), instances_per_episode=1, context_length=64):
        # There are three indices for each frame:
        # 1. Frame id given in the episode filename (and frameids in npz), which may be discontinuous and not start from 0.
        # 2. Row in the `data` array. pos2id[2]=1 & id2pos[1]=2.
        # 3. Position in the final dataset, which contains only the annotated subset of data. data_index[3]=2
        a = np.load(path, allow_pickle=True)
        data = a['data']
        fieldnames = a['fieldnames'].tolist()
        # do a little normalization:
        normalized_fields = [ fieldnames.index(f) for f in ('tcpg', 'button', 'switch') ]
        data[:,normalized_fields] = data[:,normalized_fields] * 10.0
        pos2id = a['frameids']
        id2pos = np.full(1+max(pos2id), -1)
        for (pos, id) in enumerate(pos2id):
            id2pos[id] = pos
        lang = a['lang']
        int2task = a['tasknames']
        task2int = {}
        for (i,task) in enumerate(int2task):
            task2int[task] = i
        data_index = []; target = []; frame = []
        for (start_frame, end_frame, taskid, annot) in lang:
            for instance_frame in range(end_frame - instances_per_episode + 1, end_frame+1):
                instance_index = id2pos[instance_frame]
                if instance_index - context_length + 1 >= 0:
                    frame.append(instance_frame)
                    target.append(int(taskid))
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
