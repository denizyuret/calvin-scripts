# for i in *-tactile.tsv.gz; do j=`echo $i | sed 's/tactile/tactile2/'`; echo $i $j; zcat $i | python ../calvin_fix_tactile.py | gzip > $j; done

import sys
import numpy as np

a = np.loadtxt(sys.stdin, delimiter='\t')
a[:,1:3] = a[:,1:3] * 100.0
a[:,3:9] = a[:,3:9] / 255.0
np.savetxt(sys.stdout, a, delimiter='\t', fmt='%g')

