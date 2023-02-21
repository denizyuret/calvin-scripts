# python calvin_language_task.py
# extract all unique task ids and language annotations

import numpy as np

lang = {}

calvin_dir = '/datasets/calvin/'

for d in ['debug/training/lang_annotations',
          'debug/validation/lang_annotations',
          'D/training/lang_annotations',
          'D/validation/lang_annotations',
          'ABC/training/lang_annotations',
          'ABC/validation/lang_annotations',
          'ABCD/training/lang_annotations',
          'ABCD/validation/lang_annotations']:
    f = calvin_dir + d + '/auto_lang_ann.npy'
    a = np.load(f, allow_pickle=True).item()
    for x,y in zip(a['language']['task'], a['language']['ann']):
        lang[x + '\t' + y] = 1

for x in sorted(lang.keys()):
    print(x)
