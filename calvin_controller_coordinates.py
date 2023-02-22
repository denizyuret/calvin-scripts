# zcat D-validation.tsv.gz | python calvin_controller_coordinates.py <field> | python calvin_controller_regression.py
# Utility to figure out the coordinates of controllers (buttons etc).
# When the value of <field> (column number door:30,drawer:31,button:32,switch:33) 
# changes, see what the tcp (tool center point) is doing in xyz.
# We can then feed this data to linear regression etc. to figure out coord xform.

import sys

field = int(sys.argv[1])
print(f"Observing movement in field {field}", file=sys.stderr)
field_val = ''
buffer = []

for line in sys.stdin:
    fields = line.strip().split('\t')
    f = float(fields[field])
    if field_val == '':
        field_val = f
        continue
    elif abs(f - field_val) > 0.0001:
        buffer.append(f"{fields[0]}\t{f}\t{fields[15]}\t{fields[16]}\t{fields[17]}")
        field_val = f
    else:
        if len(buffer) > 10:
            for b in buffer:
                print(b)
        buffer = []

if len(buffer) > 10:
    for b in buffer:
        print(b)
        buffer = []
