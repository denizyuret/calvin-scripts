using PyCall, DelimitedFiles
np = pyimport("numpy")

for f in readdir()
    m = match(r"episode_(\d{7})\.npz", f)
    if m != nothing
        idnum = m.captures[1]
        data = np.load(f, allow_pickle=true)
        actions = get(data, "actions") # 1-7(7)
        rel_actions = get(data, "rel_actions") # 8-14(7)
        robot_obs = get(data, "robot_obs") # 15-29(15)
        scene_obs = get(data, "scene_obs") # 30-53(24)
        print(idnum, '\t')
        writedlm(stdout, vcat(actions, rel_actions, robot_obs, scene_obs)')
    end
end

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
