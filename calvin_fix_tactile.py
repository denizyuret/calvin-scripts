# for i in *-tactile.tsv.gz; do j=`echo $i | sed 's/tactile/tactile2/'`; echo $i $j; zcat $i | python ../calvin_fix_tactile.py | gzip > $j; done

import sys
import numpy as np
from io import StringIO

for line in sys.stdin:
    fields = line.strip().split('\t')
    nfields = np.loadtxt(StringIO(line), delimiter='\t')
    nfields[1:3] = nfields[1:3] * 100.0
    nfields[3:9] = nfields[3:9] / 255.0
    print(fields[0], end='')
    for f in nfields[1:]:
        print(f"\t{f:g}", end='')
    print('')
