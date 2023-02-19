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


# | directory | scene_info.npy |
# | --------- | -------------- |
# | debug/training | {'calvin_scene_D': [358482, 361252]} |
# | debug/validation | {'calvin_scene_D': [553567, 555241]} |
# | D/training | {'calvin_scene_A': [0, 611098]} |
# | D/validation | . |
# | ABC/training | {'calvin_scene_B': [0, 598909], 'calvin_scene_C': [598910, 1191338], 'calvin_scene_A': [1191339, 1795044]} |
# | ABC/validation | . |
# | ABCD/training | {'calvin_scene_A': [1802438, 2406143], 'calvin_scene_B': [611099, 1210008], 'calvin_scene_C': [1210009, 1802437], 'calvin_scene_D': [0, 611098]} |
# | ABCD/validation | . |
