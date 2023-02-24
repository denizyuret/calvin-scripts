# python calvin_controller_check.py D-validation.tsv.gz D-validation-controller.tsv.gz
# Verifies the coordinates in D-validation-controller by comparing them with tcp coordinates in D-validation whenever a controller is moving.

import sys
import gzip
from io import StringIO
import numpy as np
import pandas as pd

prev = []
slider = []
drawer = []
button = []
switch = []
counter = 0
mincounter = 3

with gzip.open(sys.argv[1], 'rt') as orig, gzip.open(sys.argv[2], 'rt') as cont:
    for orig_line, cont_line in zip(orig, cont):
        a = np.loadtxt(StringIO(orig_line.strip()), delimiter='\t', max_rows=1)
        if len(prev) == 0:
            prev = a
            continue
        b = np.loadtxt(StringIO(cont_line.strip()), delimiter='\t', max_rows=1)
        if abs(a[30] - prev[30]) > 0.001:
            if counter > mincounter:
                slider.append(np.linalg.norm(a[15:18] - b[1:4]))
            else:
                counter = counter + 1
        elif abs(a[31] - prev[31]) > 0.001:
            if counter > mincounter:
                drawer.append(np.linalg.norm(a[15:18] - b[4:7]))
            else:
                counter = counter + 1
        elif abs(a[32] - prev[32]) > 0.001:
            if counter > mincounter:
                button.append(np.linalg.norm(a[15:18] - b[7:10]))
            else:
                counter = counter + 1
        elif abs(a[33] - prev[33]) > 0.001:
            if counter > mincounter:
                switch.append(np.linalg.norm(a[15:18] - b[10:13]))
            else:
                counter = counter + 1
        else:
            counter = 0
        prev = a

if slider:
    print('slider: ', pd.DataFrame(slider).describe())
if drawer:
    print('drawer: ', pd.DataFrame(drawer).describe())
if button:
    print('button: ', pd.DataFrame(button).describe())
if switch:
    print('switch: ', pd.DataFrame(switch).describe())

