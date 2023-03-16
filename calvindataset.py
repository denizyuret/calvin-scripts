import numpy as np
import torch
from torch.utils.data import Dataset

# Defines CalvinDataset, TwoFrameDataset.

# Features: (0, 'actx'),(1, 'acty'),(2, 'actz'),(3, 'acta'),(4, 'actb'),(5, 'actc'),(6, 'actg'),(7, 'relx'),(8, 'rely'),(9, 'relz'),(10, 'rela'),(11, 'relb'),(12, 'relc'),(13, 'relg'),(14, 'tcpx'),(15, 'tcpy'),(16, 'tcpz'),(17, 'tcpa'),(18, 'tcpb'),(19, 'tcpc'),(20, 'tcpg'),(21, 'arm1'),(22, 'arm2'),(23, 'arm3'),(24, 'arm4'),(25, 'arm5'),(26, 'arm6'),(27, 'arm7'),(28, 'armg'),(29, 'slider'),(30, 'drawer'),(31, 'button'),(32, 'switch'),(33, 'lightbulb'),(34, 'greenlight'),(35, 'redx'),(36, 'redy'),(37, 'redz'),(38, 'reda'),(39, 'redb'),(40, 'redc'),(41, 'bluex'),(42, 'bluey'),(43, 'bluez'),(44, 'bluea'),(45, 'blueb'),(46, 'bluec'),(47, 'pinkx'),(48, 'pinky'),(49, 'pinkz'),(50, 'pinka'),(51, 'pinkb'),(52, 'pinkc'),(53, 'slider.x'),(54, 'slider.y'),(55, 'slider.z'),(56, 'drawer.x'),(57, 'drawer.y'),(58, 'drawer.z'),(59, 'button.x'),(60, 'button.y'),(61, 'button.z'),(62, 'switch.x'),(63, 'switch.y'),(64, 'switch.z'),(65, 'tact1d'),(66, 'tact2d'),(67, 'tact1r'),(68, 'tact1g'),(69, 'tact1b'),(70, 'tact2r'),(71, 'tact2g'),(72, 'tact2b'),(73, 'slider_diff'),(74, 'drawer_diff'),(75, 'button_diff'),(76, 'switch_diff'),(77, 'lightbulb_diff'),(78, 'greenlight_diff'),(79, 'redx_diff'),(80, 'redy_diff'),(81, 'redz_diff'),(82, 'reda_diff'),(83, 'redb_diff'),(84, 'redc_diff'),(85, 'bluex_diff'),(86, 'bluey_diff'),(87, 'bluez_diff'),(88, 'bluea_diff'),(89, 'blueb_diff'),(90, 'bluec_diff'),(91, 'pinkx_diff'),(92, 'pinky_diff'),(93, 'pinkz_diff'),(94, 'pinka_diff'),(95, 'pinkb_diff'),(96, 'pinkc_diff')


class CalvinDataset(Dataset):
    """
    Turn the last `instances_per_episode` frames of each episode into instances 
    with a subset of `features` from the last `context_length` frames.
    """
    def __init__(self, path='data/debug-training.npz', features=range(0,97), instances_per_episode=1, context_length=64):
        # There are three indices for each frame:
        # 1. Frame id given in the episode filename (and frameids in npz), which may be discontinuous and not start from 0.
        # 2. Row in the `data` array. frameids[2]=1 & id2pos[1]=2.
        # 3. Position in the final dataset, which contains only the annotated subset of data. data_index[3]=2
        npzfile = np.load(path, allow_pickle=True)
        data = npzfile['data']
        lang = npzfile['lang']
        frameids = npzfile['frameids']
        tasknames = npzfile['tasknames'].tolist()
        fieldnames = npzfile['fieldnames'].tolist()
        episodes = npzfile['episodes']
        normalized_fields = [ fieldnames.index(f) for f in ('tcpg', 'button', 'switch') ]
                
        # build reverse maps
        id2pos = np.full(1+max(frameids), -1)
        for (pos, id) in enumerate(frameids):
            id2pos[id] = pos
        task2int = {}
        for (i,task) in enumerate(tasknames):
            task2int[task] = i

        # prepare to index only labeled data
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
        inputs = self.data[(i-self.context_length+1):(i+1),:].copy()
        inputs = self.normalize(inputs)
        inputs = inputs[:, self.features]
        inputs = torch.from_numpy(inputs)
        target = torch.tensor(self.target[index])
        frame = torch.tensor(self.frame[index])
        return (inputs, target, frame)

    def __len__(self):
        return len(self.target)

    def normalize(self, inputs):
        inputs[:,self.normalized_fields] = inputs[:,self.normalized_fields] * 10.0
        return inputs


class TwoFrameDataset(Dataset):
    """
    Gives the `features` of the last frame and last-`context_length`+1 frame for each episode.
    """
    def __init__(self, path='data/debug-training.npz', features=range(0,97), context_length=32):
        # There are three indices for each frame:
        # 1. Frame id given in the episode filename (and frameids in npz), which may be discontinuous and not start from 0.
        # 2. Row in the `data` array. frameids[2]=1 & id2pos[1]=2.
        # 3. Position in the final dataset, which contains only the annotated subset of data. data_index[3]=2
        npzfile = np.load(path, allow_pickle=True)
        data = npzfile['data']
        lang = npzfile['lang']
        frameids = npzfile['frameids']
        tasknames = npzfile['tasknames'].tolist()
        fieldnames = npzfile['fieldnames'].tolist()
        episodes = npzfile['episodes']
        normalized_fields = [ fieldnames.index(f) for f in ('tcpg', 'button', 'switch') ]
                
        # build reverse maps
        id2pos = np.full(1+max(frameids), -1)
        for (pos, id) in enumerate(frameids):
            id2pos[id] = pos
        task2int = {}
        for (i,task) in enumerate(tasknames):
            task2int[task] = i

        # prepare to index only labeled data
        data_index = []; target = []; frame = []
        for (start_frame, end_frame, taskid, annot) in lang:
            instance_frame = end_frame
            instance_index = id2pos[instance_frame]
            if instance_index - context_length + 1 >= 0:
                frame.append(instance_frame)
                target.append(int(taskid))
                data_index.append(instance_index)
        self.__dict__.update(locals()) # convert each local variable to self.var


    def __getitem__(self, index):
        i = self.data_index[index]
        inputs = self.data[[i-self.context_length+1,i],:]
        inputs = self.normalize(inputs)
        inputs = inputs[:, self.features]
        inputs = torch.from_numpy(inputs)
        target = torch.tensor(self.target[index])
        frame = torch.tensor(self.frame[index])
        return (inputs, target, frame)

    def __len__(self):
        return len(self.target)

    def normalize(self, inputs):
        inputs[:,self.normalized_fields] = inputs[:,self.normalized_fields] * 10.0
        return inputs

