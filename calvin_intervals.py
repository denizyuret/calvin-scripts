# zcat D-validation.tsv.gz | python calvin_intervals.py
# Prints out intervals based on idnum discontinuities

import sys

interval_start = -1
last_index = -1

for line in sys.stdin:
    fields = line.strip().split('\t')
    index = int(fields[0])
    if interval_start < 0:
        interval_start = last_index = index
    elif index == last_index + 1:
        last_index = index
    else:
        print(interval_start, '\t', last_index)
        interval_start = last_index = index

print(interval_start, '\t', last_index)

# $ for i in *.gz; do echo $i; zcat $i | python ../calvin_intervals.py; done

## ABCD-training.tsv.gz
#D 0037682      0037682		??? singleton, repeats in val
#D 0053818	0053818		??? 53818 repeats in val
#D 0053819	0219634		
#D 0244284	0244284		??? 244284 repeats in val
#D 0244285	0399945		
#D 0420498 	0420498		??? 420498 repeats in val
#D 0420499	0611098
#B 0611099	1210008		same as B section of ABC
#C 1210009	1802437		same as C section of ABC
#A 1802438	2406143		same as A section of ABC

## ABCD-validation.tsv.gz	all validations have the same D sections
#D 0000000 	0053818
#D 0219635 	0244284
#D 0399946 	0420498

## ABC-training.tsv.gz
#B 0000000 	0598909
#C 0598910	1191338
#A 1191338	1795044

## ABC-validation.tsv.gz
#D 0000000 	0053818
#D 0219635 	0244284
#D 0399946 	0420498

## debug-training.tsv.gz
#D 0358482 	0361252

## debug-validation.tsv.gz
#D 0553567 	0555241

## D-training.tsv.gz		??? scene-info.npy says dataset A!
#D 0053819 	0219634
#D 0244285 	0399945
#D 0420499 	0611098

## D-validation.tsv.gz
#D 0000000 	0053818
#D 0219635 	0244284
#D 0399946 	0420498
