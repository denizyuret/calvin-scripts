# python calvin_extract_language.py auto_lang_ann.npy
# extract index-range, task, instruction triples in tab separated format

import sys
import numpy as np

a = np.load(sys.argv[1], allow_pickle=True).item()
for x,y,z in zip(a['info']['indx'], a['language']['task'], a['language']['ann']):
    print(x[0], '\t', x[1], '\t', y, '\t', z)
