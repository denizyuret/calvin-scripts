# python calvin_extract_language.py auto_lang_ann.npy
# extract index-range, task, instruction triples in tab separated format
# to extract all:
# for i in debug D ABC ABCD; do for j in validation training; do k=/datasets/calvin/$i/$j/lang_annotations/auto_lang_ann.npy; python ../calvin_extract_language.py $k | gzip > $i-$j-lang.tsv.gz; done; done

import sys
import numpy as np

a = np.load(sys.argv[1], allow_pickle=True).item()
for x,y,z in zip(a['info']['indx'], a['language']['task'], a['language']['ann']):
    print(*x, y, z, sep='\t')
