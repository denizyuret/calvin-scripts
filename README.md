# calvin-scripts

* `visualize_dataset.py`: script from calvin developers. Try `pip install opencv-python` to make `import cv2` work.

* `calvin_extract_numbers.py,jl`: extracts the numeric fields of calvin data files in current directory and prints them to stdout in tab separated format. The fields are given below. Only file-id(1), actions(7), rel_actions(7), robot_obs(15), scene_obs(24) are printed (54 columns). https://github.com/mees/calvin/blob/main/dataset/README.md has more details about the splits and the fields. We also provide description and statistics in the next section.

* `debug-training.tsv`, `debug-validation.tsv`, etc.: https://github.com/denizyuret/calvin-scripts/releases/tag/v0.0.1 The release page has the output of calvin_extract_numbers scripts.

* `calvin_scene_info.py`: Read and print info in the scene_info.npy, ep_lens.npy, ep_start_end_ids.npy files in each calvin data directory.

* `calvin_extract_unique_annotations.py`: Print all the unique task_ids and language annotations in the dataset. There are 34 unique task ids and 389 unique annotations.

* `zcat D-validation.tsv.gz | python calvin_diffs.py`: prints out the differences with the previous line for all but the first line.

* `zcat D-validation.tsv.gz | python calvin_episodes.py`: Tries to guess the episode boundaries if xyz of successive frames differ by more than 8.5 std.

* `zcat D-validation.tsv.gz | python calvin_intervals.py`: Prints out intervals based on idnum discontinuities.

* `zcat D-validation.tsv.gz | python calvin_summarize_numbers.py`: Print statistics for each column.

* `python visualize_calvin_npz.py /datasets/calvin/D/validation/episode_0000000.npz`: Visualize a single frame from a given npz file.

* `python calvin_visualize.py`: Visualize a series of frames in the current directory, a more detailed version of the original visualize_dataset.py.

* `calvin_controller_coordinates.py`, `calvin_controller_regression.py`, `calvin_controller_xyz.py`, `calvin_controller_check.py`: Scripts I tried to discover, output and check the controller (button etc) coordinates with.

* `calvin_extract_tactile.py`: Extract the mean pixels of `depth_tactile` (2) and `rgb_tactile` images. Saved as e.g. `D-validation-tactile.tsv.gz`.

* `calvin_fix_tactile.py`: The initial version of extract_tactile did not normalize the values, depth pixels were too small, rgb pixels were too large. I fixed these using this script, and named the normalized files e.g. `D-validation-tactile2.tsv.gz`. The extract script is also fixed now and normalizes.

* `python calvin_extract_language.py auto_lang_ann.npy`: extract start, end, task, instruction triples in tab separated format

* `python calvin_scene_diff.py D/validation`: extract and normalize scene coordinate differences, saved in sdiff files.

## calvin-files

* episode_XXXXXXX.npz: Each frame is represented in a file named episode_idnum.npz, consecutive idnums indicate consecutive frames (with the exception of episode transitions I guess). Other files indicating the contents: 
* scene_info.npy indicates the first and last frame numbers for a particular directory as well as which scene (A,B,C,D) they come from (although there seems to be some confusion about this). It only exists in */training but describes the union of training and validation.
* ep_start_end_ids.npy indicates the start and end idnums of segments in that particular directory.
* ep_lens.npy indicates the lengths of segments given by ep_start_end_ids.npy.
* statistics.yaml gives basic stats for numeric variables.

| directory | frames | scene_info.npy |
| --------- | ------ | -------------- |
| debug/training | 2771 | {'calvin_scene_D': [358482, 361252]} |
| debug/validation | 1675 | {'calvin_scene_D': [553567, 555241]} |
| D/training | 512077 | {'calvin_scene_D': [0, 611098]} |
| D/validation | 99022 | . |
| ABC/training | 1795045 | {'calvin_scene_B': [0, 598909], 'calvin_scene_C': [598910, 1191338], 'calvin_scene_A': [1191339, 1795044]} |
| ABC/validation | 99022 | . |
| ABCD/training | 2307126 | {'calvin_scene_D': [0, 611098], 'calvin_scene_B': [611099, 1210008], 'calvin_scene_C': [1210009, 1802437], 'calvin_scene_A': [1802438, 2406143]} |
| ABCD/validation | 99022 | . |

The validation directories for ABCD, ABC, and D have the same content: calvin_scene_D:
0:53818, 219635:244284, 399946:420498. (99022 frames). However note that the language
annotations in the D split is different from the ones in the ABC/ABCD splits.

The training directory for D has the rest of the scene D data: calvin_scene_D: 53819:219634,
244285:399945, 420499:611098. (512077 frames).  (scene_info.npy says scene_A but should be
scene_D).

The training directory for ABCD starts with the same content as D/training (except for
several off-by-one errors) followed by B (598910), C (592429), and A (603706) sections which
are identical to ABC/training but renumbered.

The scenes differ by desk color and drawer positioning. The objects look the same.


## calvin-data

A summary of all data (including images) in episode_XXXXXXX.npz files (each represents a single 1/30 sec frame):

```
# julia> for f in data[:files]; println(f, "\t", summary(get(data, f))); end
# actions	7-element Vector{Float64}
# rel_actions	7-element Vector{Float64}
# robot_obs	15-element Vector{Float64}
# scene_obs	24-element Vector{Float64}
# rgb_static	200×200×3 Array{UInt8, 3}
# rgb_gripper	84×84×3 Array{UInt8, 3}
# rgb_tactile	160×120×6 Array{UInt8, 3}
# depth_static	200×200 Matrix{Float32}
# depth_gripper	84×84 Matrix{Float32}
# depth_tactile	160×120×2 Array{Float32, 3}
```

## calvin-tsv-fields

The fields in the output of calvin_extract_numbers.py (files like D-training.tsv.gz) is as follows:

00. idnum
01. actions/x (tcp (tool center point) position: x,y,z in absolute world coordinates: we want in the next frame, i.e. act[t] = tcp[t+1])
02. actions/y
03. actions/z
04. actions/a (tcp orientation (3): euler angles a,b,c in absolute world coordinates)
05. actions/b
06. actions/c
07. actions/g (gripper_action (1): binary close=-1, open=1)
08. rel_actions/x (tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50: rel[t]=normalize(act[t]-tcp[t]))
09. rel_actions/y (normalize_dist=clip(act-obs,-0.02,0.02)/0.02, normalized_angle=clip(((act-obs) + pi) % (2*pi) - pi, -0.05, 0.05)/0.05
10. rel_actions/z
11. rel_actions/a (tcp orientation (3): euler angles a,b,c in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20)
12. rel_actions/b
13. rel_actions/c
14. rel_actions/g (gripper_action (1): binary close=-1, open=1)
15. robot_obs/x (tcp position (3): x,y,z in world coordinates: current position of tcp, i.e. tcp[t]=act[t-1])
16. robot_obs/y
17. robot_obs/z
18. robot_obs/a (tcp orientation (3): euler angles a,b,c in world coordinates)
19. robot_obs/b
20. robot_obs/c
21. robot_obs/w (gripper opening width (1): in meters)
22. robot_obs/j1 (arm_joint_states (7): in rad)
23. robot_obs/j2
24. robot_obs/j3
25. robot_obs/j4
26. robot_obs/j5
27. robot_obs/j6
28. robot_obs/j7
29. robot_obs/g (gripper_action (1): binary close = -1, open = 1)
30. scene_obs/sliding_door (1): joint state: range=[-0.002359:0.306696]
31. scene_obs/drawer (1): joint state: range=[-0.002028:0.221432]
32. scene_obs/button (1): joint state: range=[-0.000935:0.033721]
33. scene_obs/switch (1): joint state: range=[-0.004783:0.091777]
34. scene_obs/lightbulb (1): on=1, off=0
35. scene_obs/green light (1): on=1, off=0
36. scene_obs/redx (red block (6): (x, y, z, euler_x, euler_y, euler_z)
37. scene_obs/redy
38. scene_obs/redz
39. scene_obs/reda
40. scene_obs/redb
41. scene_obs/redc
42. scene_obs/bluex (blue block (6): (x, y, z, euler_x, euler_y, euler_z)
43. scene_obs/bluey
44. scene_obs/bluez
45. scene_obs/bluea
46. scene_obs/blueb
47. scene_obs/bluec
48. scene_obs/pinkx (pink block (6): (x, y, z, euler_x, euler_y, euler_z)
49. scene_obs/pinky
50. scene_obs/pinkz
51. scene_obs/pinka
52. scene_obs/pinkb
53. scene_obs/pinkc

## calvin-controller-coordinates

The fields in the output of calvin_controller_xyz.py (files like D-training-controllers.tsv.gz) are as follows:
The coordinates give the location tcp should be at to move the controller at each point in time.
For details of the coordinate calculation see calvin_controller_xyz.md and calvin_controller_xyz.py.

00.00 idnum
01.54 slider.x
02.55 slider.y
03.56 slider.z
04.57 drawer.x
05.58 drawer.y
06.59 drawer.z
07.60 button.x
08.61 button.y
09.62 button.z
10.63 switch.x
11.64 switch.y
12.65 switch.z

## calvin-tactile-coordinates

The fields in the output of the calvin_extract_tactile.py script (files like `D-validation-tactile2.tsv.gz`). These give the average pixel of the tactile images. The depth pixels were normalized with x100, the rgb pixels were normalized with /255.0.

00.00 idnum
01.66 depth_tactile1
02.67 depth_tactile2
03.68 rgb_tactile1_r
04.69 rgb_tactile1_g
05.70 rgb_tactile1_b
06.71 rgb_tactile2_r
07.72 rgb_tactile2_g
08.73 rgb_tactile2_b

## calvin-scene-changes

These are the normalized differences in scene coordinates: normalize(x[next]-x[curr]).

Normalization similar to rel_actions based on https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/utils/utils.py#L160

00.00 idnum
01.74 scene_obs/sliding_door
02.75 scene_obs/drawer
03.76 scene_obs/button
04.77 scene_obs/switch
05.78 scene_obs/lightbulb
06.79 scene_obs/green_light
07.80 scene_obs/redx
08.81 scene_obs/redy
09.82 scene_obs/redz
10.83 scene_obs/reda
11.84 scene_obs/redb
12.85 scene_obs/redc
13.86 scene_obs/bluex
14.87 scene_obs/bluey
15.88 scene_obs/bluez
16.89 scene_obs/bluea
17.90 scene_obs/blueb
18.91 scene_obs/bluec
19.92 scene_obs/pinkx
20.93 scene_obs/pinky
21.94 scene_obs/pinkz
22.95 scene_obs/pinka
23.96 scene_obs/pinkb
24.97 scene_obs/pinkc


## calvin-annotation-statistics

Each language annotation file is class (task) balanced, roughly equal number of each class
for each dataset.  ABC-validation and ABCD-validation are buggy and should not be used: they
padded the annotation numbers by repeating the same annotations multiple times. Use
D-validation instead.

| dataset | % frames annotated | frames | annots | contig tasks | frames/annot | % frames multiply annotated |
| --------| ------------------ | ------ | ------ | ------------ | ------------ | --------------------------- |
| ABCD-training    | .3734 | 2307126 | 22966 | 14176 | 100.46 | .3650 |
| ABC-training     | .3760 | 1795045 | 17870 | 11102 | 100.45 | .3608 |
| D-training       | .3813 |  512077 |  5124 |  3251 |  99.94 | .3528 |
| debug-training   | .1364 |    2771 |     9 |     7 | 307.89 | .3306 |
| ABCD-validation  | .0634 |   99022 |  1087 |    90 |  91.10 | .9154 |
| ABC-validation   | .0634 |   99022 |  1087 |    90 |  91.10 | .9154 |
| D-validation     | .3660 |   99022 |  1011 |   605 |  97.94 | .3755 |
| debug-validation | .2036 |    1675 |     8 |     6 | 209.38 | .1994 |
