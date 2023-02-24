# zcat D-validation.tsv.gz | python calvin_controller_xyz.py | gzip > D-validation-controllers.tsv.gz
# Utility to convert from 1-DOF controller state to tcp coordinates.
# Outputs idnum and 12 xyz coordinates, 13 columns total.
# (one xyz for each of the input column numbers door:30,drawer:31,button:32,switch:33)

import sys

ABCtraining = ''

for line in sys.stdin:
    fields = line.strip().split('\t')
    idnum = fields[0]
    index = int(idnum)
    if ABCtraining == '':
        ABCtraining = ((fields[0] == '0000000' and fields[1] == '0.13506961919170005') or # ABC-training.tsv.gz B.tsv.gz
                       (fields[0] == '1191339' and fields[1] == '-0.09070799374373786') or # A.tsv.gz
                       (fields[0] == '0598910' and fields[1] == '-0.025909095434079727'))   # C.tsv.gz
        print(f"ABCtraining={ABCtraining}", file=sys.stderr)
    if ABCtraining:
        if index <= 598909:
            scene = 'B'
        elif index <= 1191338:
            scene = 'C'
        else:
            scene = 'A'
    else:
        if index <= 611098:
            scene = 'D'
        elif index <= 1210008:
            scene = 'B'
        elif index <= 1802437:
            scene = 'C'
        else:
            scene = 'A'
    slider = float(fields[30])
    drawer = float(fields[31])
    button = float(fields[32])
    switch = float(fields[33])
    if scene == 'A':
        sliderxyz = [0.04-slider, 0.00, 0.53]
        drawerxyz = [0.10, -0.20-drawer, 0.36]
        buttonxyz = [-0.28, -0.10, 0.5158 - 1.7591*button]
        switchxyz = [0.30, 0.3413*switch + 0.0211, 0.5470*switch + 0.5410]
    elif scene == 'B':
        sliderxyz = [0.23-slider, 0.00, 0.53]
        drawerxyz = [0.18, -0.20-drawer, 0.36]
        buttonxyz = [0.28, -0.12, 0.5158 - 1.7591*button]
        switchxyz = [-0.32, 0.3413*switch + 0.0211, 0.5470*switch + 0.5410]
    elif scene == 'C':
        sliderxyz = [0.20-slider, 0.00, 0.53]
        drawerxyz = [0.10, -0.20-drawer, 0.36]
        buttonxyz = [-0.12, -0.12, 0.5158 - 1.7591*button]
        switchxyz = [-0.32, 0.3413*switch + 0.0211, 0.5470*switch + 0.5410]
    elif scene == 'D':
        sliderxyz = [0.04-slider, 0.00, 0.53]
        drawerxyz = [0.18, -0.20-drawer, 0.36]
        buttonxyz = [-0.12, -0.12, 0.5158 - 1.7591*button]
        switchxyz = [0.30, 0.3413*switch + 0.0211, 0.5470*switch + 0.5410]
    print(idnum, end='')
    for x in sliderxyz + drawerxyz + buttonxyz + switchxyz:
        print(f"\t{x:.4g}", end='')
    print('')
