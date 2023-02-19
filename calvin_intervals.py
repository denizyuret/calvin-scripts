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
# ABCD-training.tsv.gz
# 37682 	 37682		??? singleton, repeats in val
# 53818 	 219634		??? 53818 repeats in val
# 244284 	 399945		??? 244284 repeats in val
# 420498 	 1992529	??? 420498 repeats in val
# ABCD-validation.tsv.gz
#D 0 	 53818
#D 219635 	 244284
#D 399946 	 420498
# ABC-training.tsv.gz
# 0 	 1795044
## ABC-validation.tsv.gz
#D 0 	 53818
#D 219635 	 244284
#D 399946 	 420498
## debug-training.tsv.gz
#D 358482 	 361252
## debug-validation.tsv.gz
#D 553567 	 555241
## D-training.tsv.gz		??? scene-info.npy says dataset A!
#D 53819 	 219634
#D 244285 	 399945
#D 420499 	 611098
## D-validation.tsv.gz
#D 0 	 53818
#D 219635 	 244284
#D 399946 	 420498
