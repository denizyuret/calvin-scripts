# Usage: cd /datasets/calvin/D/validation; python calvin_visualize.py

import sys
import os
import re
import pdb
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from tqdm import tqdm
root = tk.Tk()

def float2uint8(a):
    return (255 * (a - a.min()) / (a.max() - a.min() + sys.float_info.epsilon)).astype(np.uint8)

def array2image(a):
    if np.issubdtype(a.dtype, np.floating):
        a = float2uint8(a)
    return ImageTk.PhotoImage(Image.fromarray(a))

def numeric_fields(npz, idnum):
    index = int(idnum)
    scene = ''
    if scenes:
        for (key, (low, high)) in scenes.items():
            if index >= low and index <= high:
                scene = key[-1]
                break
    ep_start = ''
    ep_end = ''
    if episodes:
        for (low, high) in episodes:
            if index >= low and index <= high:
                ep_start = low
                ep_end = high
                break
    indexstr = f"frame: {idnum} ({scene}:{ep_start}:{ep_end})"
    
    a = npz['actions']
    actions = f"act:  x:{a[0]: 5.2f} y:{a[1]: 5.2f} z:{a[2]: 5.2f} a:{a[3]: 5.2f} b:{a[4]: 5.2f} c:{a[5]: 5.2f} grp:{a[6]: 5.2f}"
    b = npz['rel_actions']
    rel_actions = f"rel:  x:{b[0]: 5.2f} y:{b[1]: 5.2f} z:{b[2]: 5.2f} a:{b[3]: 5.2f} b:{b[4]: 5.2f} c:{b[5]: 5.2f} grp:{b[6]: 5.2f}"
    c = npz['robot_obs']
    robot_obs = f"tcp:  x:{c[0]: 5.2f} y:{c[1]: 5.2f} z:{c[2]: 5.2f} a:{c[3]: 5.2f} b:{c[4]: 5.2f} c:{c[5]: 5.2f} grp:{c[6]*100: 5.2f}"
    robot_arm = f"arm:  a:{c[7]: 5.2f} b:{c[8]: 5.2f} c:{c[9]: 5.2f} d:{c[10]: 5.2f} e:{c[11]: 5.2f} f:{c[12]: 5.2f} g:{c[13]: 5.2f} grp:{c[14]: 5.2f}"
    d = npz['scene_obs']
    red = f"red:  x:{d[6]: 5.2f} y:{d[7]: 5.2f} z:{d[8]: 5.2f} a:{d[9]: 5.2f} b:{d[10]: 5.2f} c:{d[11]: 5.2f}"
    blue = f"blue: x:{d[12]: 5.2f} y:{d[13]: 5.2f} z:{d[14]: 5.2f} a:{d[15]: 5.2f} b:{d[16]: 5.2f} c:{d[17]: 5.2f}"
    pink = f"pink: x:{d[18]: 5.2f} y:{d[19]: 5.2f} z:{d[20]: 5.2f} a:{d[21]: 5.2f} b:{d[22]: 5.2f} c:{d[23]: 5.2f}"
    desk = f"door:{d[0]: 5.2f} drawer:{d[1]: 5.2f} button:{d[2]: 5.2f} switch:{d[3]: 5.2f} bulb:{d[4]: 5.2f} green:{d[5]: 5.2f}"
    ann = []
    prev = ''
    if annotations:
        for n, ((low, high), t, s) in enumerate(annotations):
            if index > high:
                prev = f"<{low}:{high}:{t}: {s}"
            elif index >= low and index <= high:
                if not ann and prev:
                    ann.append(prev)
                ann.append(f"={low}:{high}:{t}: {s}")
            elif index < low:
                if not ann and prev:
                    ann.append(prev)
                ann.append(f">{low}:{high}:{t}: {s}")
                break
            
    return '\n'.join((indexstr, actions, rel_actions, robot_obs, robot_arm, red, blue, pink, desk, *ann))


def on_entry(event):
    index = int(entry.get())
    idnum = f"{index:07d}"
    if idnum in iddict:
        update_frame(idnum)
        idpos = iddict[idnum]
        scale.set(idpos)

def on_scale(value):
    index = int(value)
    idnum = idnums[index]
    entry.delete(0, tk.END)
    entry.insert(0, idnum)
    update_frame(idnum)

def update_frame(idnum):
    npz = np.load(f"episode_{idnum}.npz", allow_pickle=True)
    tklabels[1].image = array2image(npz['rgb_static'])
    tklabels[1].config(image = tklabels[1].image)
    tklabels[3].image = array2image(npz['depth_static'])
    tklabels[3].config(image = tklabels[3].image)
    tklabels[5].image = array2image(npz['rgb_gripper'])
    tklabels[5].config(image = tklabels[5].image)
    tklabels[7].image = array2image(npz['depth_gripper'])
    tklabels[7].config(image = tklabels[7].image)
    tklabels[9].image = array2image(npz['rgb_tactile'][:,:,0:3])
    tklabels[9].config(image = tklabels[9].image)
    tklabels[11].image = array2image(npz['rgb_tactile'][:,:,3:6])
    tklabels[11].config(image = tklabels[11].image)
    tklabels[13].image = array2image(npz['depth_tactile'][:,:,0])
    tklabels[13].config(image = tklabels[13].image)
    tklabels[15].image = array2image(npz['depth_tactile'][:,:,1])
    tklabels[15].config(image = tklabels[15].image)
    tklabels[16].text = numeric_fields(npz, idnum)
    tklabels[16].config(text = tklabels[16].text)


# Read filenames:
idnums = []
iddict = {}
for f in tqdm(sorted(os.listdir('.'))):
    m = re.match(r"episode_(\d{7})\.npz", f)
    if m is not None:
        idnum = m.group(1)
        iddict[idnum] = len(idnums)
        idnums.append(idnum)



# Read annotations:
if os.path.exists('lang_annotations/auto_lang_ann.npy'):
    annotations = np.load('lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
    annotations = sorted(list(zip(annotations['info']['indx'], annotations['language']['task'], annotations['language']['ann'])))
else:
    print('lang_annotations/auto_lang_ann.npy does not exist, annotations will not be displayed', file=sys.stderr)
    annotations = false


# Read episode boundaries:
if os.path.exists('ep_start_end_ids.npy'):
    episodes = sorted(np.load('ep_start_end_ids.npy', allow_pickle=True).tolist())
else:
    print('ep_start_end_ids.npy does not exist, episode boundaries will not be displayed', file=sys.stderr)
    episodes = false


# Read scene info:
if os.path.exists('scene_info.npy'):
    scenes = np.load('scene_info.npy', allow_pickle=True).item()
else:
    print('scene_info.npy does not exist, scene ids will not be displayed', file=sys.stderr)
    scenes = False


# Arrange the windows:
row = [ tk.Frame(root) for i in range(4) ]
frm = [ [ tk.Frame(row[0]) for i in range(3) ],
        [ tk.Frame(row[1]) for i in range(4) ],
        [ tk.Frame(row[2]) for i in range(1) ],
        [ tk.Frame(row[3]) for i in range(1) ] ]
for i in range(4):
    row[i].pack()
    for j in range(len(frm[i])):
        frm[i][j].pack(side='left')

tklabels = [
    tk.Label(frm[0][0], text='rgb_static'),
    tk.Label(frm[0][0]), #, image=rgb_static),
    tk.Label(frm[0][1], text='depth_static'),
    tk.Label(frm[0][1]), # , image=depth_static),
    tk.Label(frm[0][2], text='rgb_gripper'),
    tk.Label(frm[0][2]), # , image=rgb_gripper),
    tk.Label(frm[0][2], text='depth_gripper'),
    tk.Label(frm[0][2]), # , image=depth_gripper),
    tk.Label(frm[1][0], text='rgb_tactile1'),
    tk.Label(frm[1][0]), # , image=rgb_tactile1),
    tk.Label(frm[1][1], text='rgb_tactile2'),
    tk.Label(frm[1][1]), # , image=rgb_tactile2),
    tk.Label(frm[1][2], text='depth_tactile1'),
    tk.Label(frm[1][2]), # , image=depth_tactile1),
    tk.Label(frm[1][3], text='depth_tactile2'),
    tk.Label(frm[1][3]), # , image=depth_tactile2),
    tk.Label(frm[3][0], text='', justify='left', anchor='nw')
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

# Initialize with first frame
scale.set(0)
entry.insert(0,idnums[0])
update_frame(idnums[0])

# Start the main event loop
tk.mainloop()
