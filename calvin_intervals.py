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
# ABCD-validation.tsv.gz
# 0 	 53818
# 219635 	 244284
# 399946 	 420498
# ABC-training.tsv.gz
# 0 	 1795044
# ABC-validation.tsv.gz
# 0 	 53818
# 219635 	 244284
# 399946 	 420498
# debug-training.tsv.gz
# 358482 	 361252
# debug-validation.tsv.gz
# 553567 	 555241
# D-training.tsv.gz
# 53819 	 219634
# 244285 	 399945
# 420499 	 611098
# D-validation.tsv.gz
# 0 	 53818
# 219635 	 244284
# 399946 	 420498
