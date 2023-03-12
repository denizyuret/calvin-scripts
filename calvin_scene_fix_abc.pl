# zcat ABC-training.tsv.gz | python calvin_scene_fix.py | gzip > ABC-fixed-training.tsv.gz
# ABC/training:{'calvin_scene_B': [0, 598909], 'calvin_scene_C': [598910, 1191338], 'calvin_scene_A': [1191339, 1795044]}
# ABCD/training:{'calvin_scene_D': [0, 611098], 'calvin_scene_B': [611099, 1210008], 'calvin_scene_C': [1210009, 1802437], 'calvin_scene_A': [1802438, 2406143]}
# All scene_A data has red-pink switched
# All scene_C data has red-blue switched
# red=36:41, blue=42:47, pink=48:53

while(<>) {
    my @a = split;
    if (598910 <= $a[0] && $a[0] <= 1191338) { # scene_C, swap red-blue
	@a[36..41,42..47] = @a[42..47,36..41];
	print(join("\t", @a)); print("\n");
    } elsif ($a[0] >= 1191339) { # scene_A, swap red-pink
	@a[36..41,48..53] = @a[48..53,36..41];
	print(join("\t", @a)); print("\n");
    } else {			# scene_B, no swap
	print;
    }
}
