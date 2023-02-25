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

* `python calvin_extract_language.py auto_lang_ann.npy`: extract start, end, task, instruction triples in tab separated format


## calvin-files

* episode_XXXXXXX.npz: Each frame is represented in a file named episode_idnum.npz, consecutive idnums indicate consecutive frames (with the exception of episode transitions I guess). Other files indicating the contents: 
* scene_info.npy indicates the first and last frame numbers for a particular directory as well as which scene (A,B,C,D) they come from (although there seems to be some confusion about this). It only exists in */training but describes the union of training and validation.
* ep_start_end_ids.npy indicates the start and end idnums of segments in that particular directory.
* ep_lens.npy indicates the lengths of segments given by ep_start_end_ids.npy.
* statistics.yaml gives basic stats for numeric variables.

| directory | scene_info.npy |
| --------- | -------------- |
| debug/training | {'calvin_scene_D': [358482, 361252]} |
| debug/validation | {'calvin_scene_D': [553567, 555241]} |
| D/training | {'calvin_scene_D': [0, 611098]} |
| D/validation | . |
| ABC/training | {'calvin_scene_B': [0, 598909], 'calvin_scene_C': [598910, 1191338], 'calvin_scene_A': [1191339, 1795044]} |
| ABC/validation | . |
| ABCD/training | {'calvin_scene_D': [0, 611098], 'calvin_scene_B': [611099, 1210008], 'calvin_scene_C': [1210009, 1802437], 'calvin_scene_A': [1802438, 2406143]} |
| ABCD/validation | . |

The validation directories for ABCD, ABC, and D have the same content: calvin_scene_D:
0:53818, 219635:244284, 399946:420498. (99022 frames).

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
01. actions/x (tcp (tool center point) position (3): x,y,z in absolute world coordinates)
02. actions/y
03. actions/z
04. actions/a (tcp orientation (3): euler angles a,b,c in absolute world coordinates)
05. actions/b
06. actions/c
07. actions/g (gripper_action (1): binary close=-1, open=1)
08. rel_actions/x (tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50)
09. rel_actions/y
10. rel_actions/z
11. rel_actions/a (tcp orientation (3): euler angles a,b,c in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20)
12. rel_actions/b
13. rel_actions/c
14. rel_actions/g (gripper_action (1): binary close=-1, open=1)
15. robot_obs/x (tcp position (3): x,y,z in world coordinates)
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

00. idnum
01. slider.x
02. slider.y
03. slider.z
04. drawer.x
05. drawer.y
06. drawer.z
07. button.x
08. button.y
09. button.z
10. switch.x
11. switch.y
12. switch.z


## calvin-controller-coordinates

The fields in the output of the calvin_extract_tactile.py script (files like `D-validation-tactile.tsv.gz`):

00. idnum
01. depth_tactile1
02. depth_tactile2
03. rgb_tactile1_r
04. rgb_tactile1_g
05. rgb_tactile1_b
06. rgb_tactile2_r
07. rgb_tactile2_g
08. rgb_tactile2_b
