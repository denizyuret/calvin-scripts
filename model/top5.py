import sys
import torch
from torch.nn.functional import softmax
from loaddata import int2task

def top5(model, dataset):
    (x,y,idx) = dataset.tensors
    yhat = model(x)
    prob = softmax(yhat, dim=1)
    with open('top5.out', 'w') as f:
        for i in range(len(idx)):
            print(idx[i].item(), end='', file=f)
            iprob = prob[i,:]
            (_, taskids) = torch.sort(iprob, descending=True)
            for j in range(5):
                k = taskids[j]
                print(f"\t{iprob[k].item():.4f}={int2task[k]}", end='', file=f)
            print('', file=f)
