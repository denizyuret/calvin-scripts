# calvin-scripts

* visualize_dataset.py: script from calvin developers. Try `pip install opencv-python` to make `import cv2` work.

* calvin_extract_numbers.py,jl: extracts the numeric fields of calvin data files in current directory and prints them to stdout in tab separated format. The fields are given below. Only file-id(1), actions(7), rel_actions(7), robot_obs(15), scene_obs(24) are printed (54 columns). https://github.com/mees/calvin/blob/main/dataset/README.md has more details about the splits and the fields.

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
