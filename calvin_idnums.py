import os
import numpy as np

for s in ['debug/training','debug/validation','D/training','D/validation','ABC/training','ABC/validation','ABCD/training','ABCD/validation']:
    d = '/datasets/calvin/' + s
    print('| ', end='')
    for f in ['scene_info.npy','ep_lens.npy','ep_start_end_ids.npy']:
        p = d + '/' + f
        if os.path.isfile(p):
            print(np.load(p, allow_pickle=True), end=' ')
        else:
            print('.', end='')
        print(' | ', end='')
    print('')
