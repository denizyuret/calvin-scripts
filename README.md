# calvin-scripts

* visualize_dataset.py: script from calvin developers. Try `pip install opencv-python` to make `import cv2` work.

* calvin_extract_numbers.py,jl: extracts the numeric fields of calvin data files in current directory and prints them to stdout in tab separated format. The fields are given below. Only file-id(1), actions(7), rel_actions(7), robot_obs(15), scene_obs(24) are printed (54 columns). https://github.com/mees/calvin/blob/main/dataset/README.md has more details about the splits and the fields. We also provide description and statistics in the next section.

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

* debug-training.tsv, debug-validation.tsv, etc.: https://github.com/denizyuret/calvin-scripts/releases/tag/v0.0.1 The release page has the output of calvin_extract_numbers scripts.

* calvin_idnums.py: Extract episode_idnum ranges for each directory.

# calvin-files

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
