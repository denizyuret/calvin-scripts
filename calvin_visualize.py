# Usage: python calvin_visualize.py -d /datasets/calvin/D/validation -t /tmp/top5.out

import sys
import os
import re
import pdb
import tkinter as tk
import numpy as np
from argparse import ArgumentParser
from PIL import Image, ImageTk
from tqdm import tqdm

def setup_windows(root):
    """Arrange the user interface"""
    row = [ tk.Frame(root) for i in range(4) ]
    frm = [ [ tk.Frame(row[0]) for i in range(3) ],
            [ tk.Frame(row[1]) for i in range(4) ],
            [ tk.Frame(row[2]) for i in range(1) ],
            [ tk.Frame(row[3]) for i in range(1) ] ]
    for i in range(4):
        row[i].pack()
        for j in range(len(frm[i])):
            frm[i][j].pack(side="left")

    tklabels = [
        tk.Label(frm[0][0], text="rgb_static"),
        tk.Label(frm[0][0]), #, image=rgb_static),
        tk.Label(frm[0][1], text="depth_static"),
        tk.Label(frm[0][1]), # , image=depth_static),
        tk.Label(frm[0][2], text="rgb_gripper"),
        tk.Label(frm[0][2]), # , image=rgb_gripper),
        tk.Label(frm[0][2], text="depth_gripper"),
        tk.Label(frm[0][2]), # , image=depth_gripper),
        tk.Label(frm[1][0], text="rgb_tactile1"),
        tk.Label(frm[1][0]), # , image=rgb_tactile1),
        tk.Label(frm[1][1], text="rgb_tactile2"),
        tk.Label(frm[1][1]), # , image=rgb_tactile2),
        tk.Label(frm[1][2], text="depth_tactile1"),
        tk.Label(frm[1][2]), # , image=depth_tactile1),
        tk.Label(frm[1][3], text="depth_tactile2"),
        tk.Label(frm[1][3]), # , image=depth_tactile2),
        tk.Label(frm[3][0], text="", justify="left", anchor="nw")
    ]

    for lbl in tklabels:
        lbl.pack()

    # Entry box:
    entry = tk.Entry(frm[2][0], width=7)
    entry.bind("<Return>", on_entry)
    entry.pack(side="left")

    # Scale:
    scale = tk.Scale(frm[2][0], from_=0, to=len(idnums)-1, orient=tk.HORIZONTAL, command=on_scale, length=430, resolution=1.0, showvalue=False)
    scale.pack(side="left")

    return tklabels, entry, scale
    

def float2uint8(a):
    return (255 * (a - a.min()) / (a.max() - a.min() + sys.float_info.epsilon)).astype(np.uint8)


def array2image(a):
    if np.issubdtype(a.dtype, np.floating):
        a = float2uint8(a)
    return ImageTk.PhotoImage(Image.fromarray(a))


def numeric_fields(npz, idnum):
    """Format numerical fields in npz and text as a single string"""
    index = int(idnum)
    scene = ""
    if scenes is not None:
        for (key, (low, high)) in scenes.items():
            if index >= low and index <= high:
                scene = key[-1]
                break
    ep_start = ""
    ep_end = ""
    if episodes is not None:
        for (low, high) in episodes:
            if index >= low and index <= high:
                ep_start = low
                ep_end = high
                break
    indexstr = f"frame: {idnum} ({scene}:{ep_start}:{ep_end})"
    
    a = npz["actions"]
    actions = f"act:  x:{a[0]: 5.2f} y:{a[1]: 5.2f} z:{a[2]: 5.2f} a:{a[3]: 5.2f} b:{a[4]: 5.2f} c:{a[5]: 5.2f} grp:{a[6]: 5.2f}"
    b = npz["rel_actions"]
    rel_actions = f"rel:  x:{b[0]: 5.2f} y:{b[1]: 5.2f} z:{b[2]: 5.2f} a:{b[3]: 5.2f} b:{b[4]: 5.2f} c:{b[5]: 5.2f} grp:{b[6]: 5.2f}"
    c = npz["robot_obs"]
    robot_obs = f"tcp:  x:{c[0]: 5.2f} y:{c[1]: 5.2f} z:{c[2]: 5.2f} a:{c[3]: 5.2f} b:{c[4]: 5.2f} c:{c[5]: 5.2f} grp:{c[6]*100: 5.2f}"
    robot_arm = f"arm:  a:{c[7]: 5.2f} b:{c[8]: 5.2f} c:{c[9]: 5.2f} d:{c[10]: 5.2f} e:{c[11]: 5.2f} f:{c[12]: 5.2f} g:{c[13]: 5.2f} grp:{c[14]: 5.2f}"
    d = npz["scene_obs"]
    red = f"red:  x:{d[6]: 5.2f} y:{d[7]: 5.2f} z:{d[8]: 5.2f} a:{d[9]: 5.2f} b:{d[10]: 5.2f} c:{d[11]: 5.2f}"
    blue = f"blue: x:{d[12]: 5.2f} y:{d[13]: 5.2f} z:{d[14]: 5.2f} a:{d[15]: 5.2f} b:{d[16]: 5.2f} c:{d[17]: 5.2f}"
    pink = f"pink: x:{d[18]: 5.2f} y:{d[19]: 5.2f} z:{d[20]: 5.2f} a:{d[21]: 5.2f} b:{d[22]: 5.2f} c:{d[23]: 5.2f}"
    desk = f"door:{d[0]: 5.2f} drawer:{d[1]: 5.2f} button:{d[2]: 5.2f} switch:{d[3]: 5.2f} bulb:{d[4]: 5.2f} green:{d[5]: 5.2f}"
    ann = []
    prev = ""
    if annotations is not None:
        curr_tasks = {}
        for n, ((low, high), t, s) in enumerate(annotations):
            if index > high:
                prev = f"<{low}:{high}:{t}: {s}"
            elif index >= low and index <= high:
                if not ann and prev:
                    ann.append(prev)
                if t not in curr_tasks:
                    ann.append(f"={low}:{high}:{t}: {s}")
                    curr_tasks[t] = True
            elif index < low:
                if not ann and prev:
                    ann.append(prev)
                ann.append(f">{low}:{high}:{t}: {s}")
                break

    if text is not None and index in text:
        ann.extend(text[index])

    return "\n".join((indexstr, actions, rel_actions, robot_obs, robot_arm, red, blue, pink, desk, *ann))


def on_entry(event):
    """This gets called when user enters a number"""
    index = int(entry.get())
    idnum = f"{index:07d}"
    if idnum in iddict:
        update_frame(idnum)
        idpos = iddict[idnum]
        scale.set(idpos)


def on_scale(value):
    """This gets called when user moves the slider"""
    index = int(value)
    idnum = idnums[index]
    entry.delete(0, tk.END)
    entry.insert(0, idnum)
    update_frame(idnum)


def update_frame(idnum):
    """This gets called when user changes the frame"""
    npz = np.load(f"{args.dir}/episode_{idnum}.npz", allow_pickle=True)
    tklabels[1].image = array2image(npz["rgb_static"])
    tklabels[1].config(image = tklabels[1].image)
    tklabels[3].image = array2image(npz["depth_static"])
    tklabels[3].config(image = tklabels[3].image)
    tklabels[5].image = array2image(npz["rgb_gripper"])
    tklabels[5].config(image = tklabels[5].image)
    tklabels[7].image = array2image(npz["depth_gripper"])
    tklabels[7].config(image = tklabels[7].image)
    tklabels[9].image = array2image(npz["rgb_tactile"][:,:,0:3])
    tklabels[9].config(image = tklabels[9].image)
    tklabels[11].image = array2image(npz["rgb_tactile"][:,:,3:6])
    tklabels[11].config(image = tklabels[11].image)
    tklabels[13].image = array2image(npz["depth_tactile"][:,:,0])
    tklabels[13].config(image = tklabels[13].image)
    tklabels[15].image = array2image(npz["depth_tactile"][:,:,1])
    tklabels[15].config(image = tklabels[15].image)
    tklabels[16].text = numeric_fields(npz, idnum)
    tklabels[16].config(text = tklabels[16].text)


def read_annotations(dir):
    """Read annotations"""
    annotfile = dir + "/lang_annotations/auto_lang_ann.npy"
    if os.path.exists(annotfile):
        print(f"Reading {annotfile}", file=sys.stderr)
        annotations = np.load(annotfile, allow_pickle=True).item()
        annotations = sorted(list(zip(annotations["info"]["indx"], annotations["language"]["task"], annotations["language"]["ann"])))
        print(f"Found {len(annotations)} annotations", file=sys.stderr)
    else:
        print(f"{annotfile} does not exist, annotations will not be displayed", file=sys.stderr)
        annotations = None
    return annotations


def read_episodes(dir):
    """Read episode boundaries"""
    episodefile = dir + "/ep_start_end_ids.npy"
    if os.path.exists(episodefile):
        print(f"Reading {episodefile}", file=sys.stderr)
        episodes = sorted(np.load(episodefile, allow_pickle=True).tolist())
        print(f"Found {len(episodes)} episodes")
    else:
        print(f"{episodefile} does not exist, episode boundaries will not be displayed", file=sys.stderr)
        episodes = None
    return episodes


def read_scenes(dir):
    """Read scene info"""
    scenefile = dir + "/scene_info.npy"
    if os.path.exists(scenefile):
        print(f"Reading {scenefile}", file=sys.stderr)
        scenes = np.load(scenefile, allow_pickle=True).item()
    else:
        print(f"{scenefile} does not exist, scene ids will not be displayed", file=sys.stderr)
        scenes = None
    return scenes


def read_text(path):
    """Read additional text to display, e.g. predictions. Should be tsv with first col = frame id"""
    if os.path.exists(path):
        print(f"Reading {path}", file=sys.stderr)
        text = {}
        with open(path, "rt") as f:
            for line in f:
                cols = line.split("\t")
                id = int(cols[0])
                if id in text:
                    text[id].extend(cols[1:])
                else:
                    text[id] = cols[1:]
    else:
        print(f"{path} does not exist, predictions will not be displayed", file=sys.stderr)
        text = None
    return text


def read_dir(dir):
    """Find episode-XXXXXXX.npz files in dir and return their ids"""
    idnums = []
    iddict = {}
    print(f"Reading directory {dir}", file=sys.stderr)
    for f in sorted(os.listdir(dir)):
        m = re.match(r"episode_(\d{7})\.npz", f)
        if m is not None:
            idnum = m.group(1)
            iddict[idnum] = len(idnums)
            idnums.append(idnum)
    print(f"Found {len(idnums)} frames.", file=sys.stderr)
    return idnums, iddict



if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument("-d", "--dir", default=".", type=str, help="Path to dir containing episode_XXXXXXX.npz files")
    parser.add_argument("-t", "--text", type=str, help="Path to tsv file containing additional text, (1st column=frame id).")
    args = parser.parse_args()

    idnums, iddict = read_dir(args.dir)
    if len(idnums) == 0:
        sys.exit(f"Error: Could not find any episode files in {args.dir}.")
    annotations = read_annotations(args.dir)
    episodes = read_episodes(args.dir)
    scenes = read_scenes(args.dir)
    if args.text is not None:
        text = read_text(args.text)
    else:
        text = None

    root = tk.Tk()
    (tklabels, entry, scale) = setup_windows(root)
    scale.set(0)
    entry.insert(0,idnums[0])
    update_frame(idnums[0])
    tk.mainloop()
