import sys
import torch
import loaddata as ld
import torch
from torch.nn.functional import softmax
from loaddata import int2task
import rnn
import mlp

def top5(model, prefix='../data/D-validation', features=range(1,74)):
    model.eval()
    (data,pos2id,id2pos) = ld.loaddata(prefix)
    x = torch.tensor(data[:,features])
    if isinstance(model, rnn.LitRNN):
        x = x.reshape((1, *x.shape))
    y = model(x)
    if isinstance(model, rnn.LitRNN):
        y = y.reshape(*y.shape[1:])
    prob = softmax(y, dim=1)
    with open('top5.out', 'w') as f:
        for i in range(len(pos2id)):
            print(pos2id[i], end='', file=f)
            iprob = prob[i,:]
            (_, taskids) = torch.sort(iprob, descending=True)
            for j in range(5):
                k = taskids[j]
                print(f"\t{iprob[k].item():.4f}={int2task[k]}", end='', file=f)
            print('', file=f)
