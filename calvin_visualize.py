import sys
import os
import re
import pdb
import numpy as np
from PIL import Image
from tqdm import tqdm
import gradio as gr

def float2uint8(a):
    return (255 * (a - a.min()) / (a.max() - a.min() + sys.float_info.epsilon)).astype(np.uint8)

def array2image(a):
    if np.issubdtype(a.dtype, np.floating):
        a = float2uint8(a)
    return Image.fromarray(a)

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


# Read filenames:
idnums = []
iddict = {}
for f in tqdm(sorted(os.listdir('.'))):
    m = re.match(r"episode_(\d{7})\.npz", f)
    if m is not None:
        idnum = m.group(1)
        iddict[idnum] = len(idnums)
        idnums.append(idnum)


if os.path.exists('lang_annotations/auto_lang_ann.npy'):
    annotations = np.load('lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
    annotations = sorted(list(zip(annotations['info']['indx'], annotations['language']['task'], annotations['language']['ann'])))
else:
    print('lang_annotations/auto_lang_ann.npy does not exist, annotations will not be displayed', file=sys.stderr)
    annotations = False


# Read episode boundaries:
if os.path.exists('ep_start_end_ids.npy'):
    episodes = sorted(np.load('ep_start_end_ids.npy', allow_pickle=True).tolist())
else:
    print('ep_start_end_ids.npy does not exist, episode boundaries will not be displayed', file=sys.stderr)
    episodes = False


# Read scene info:
if os.path.exists('scene_info.npy'):
    scenes = np.load('scene_info.npy', allow_pickle=True).item()
else:
    print('scene_info.npy does not exist, scene ids will not be displayed', file=sys.stderr)
    scenes = False


with gr.Blocks() as demo:
    with gr.Row():
        rgb_static = gr.Image(interactive=False,label="rgb_static")
        depth_static = gr.Image(interactive=False,label="depth_static")
        rgb_gripper = gr.Image(interactive=False,label="rgb_gripper")
        depth_gripper = gr.Image(interactive=False,label="depth_gripper")

    with gr.Row():
        rgb_tactile1 = gr.Image(interactive=False,label="rgb_tactile1")
        rgb_tactile2 = gr.Image(interactive=False,label="rgb_tactile2")
        depth_tactile1 = gr.Image(interactive=False,label="depth_tactile1")
        depth_tactile2 = gr.Image(interactive=False,label="depth_tactile2")
    
    with gr.Row():
        text_info = gr.Text(label="text")


    with gr.Row():
        slider = gr.Slider(label="slider", minimum=0, maximum=len(idnums)-1, step=1, value=0)
 
    def update_frame(value):
        index = int(value)
        idnum = idnums[index]
        npz = np.load(f"episode_{idnum}.npz", allow_pickle=True)
        return (array2image(npz['rgb_static']), 
                array2image(npz['depth_static']),
                array2image(npz['rgb_gripper']), 
                array2image(npz['depth_gripper']),
                array2image(npz['rgb_tactile'][:,:,0:3]),
                array2image(npz['rgb_tactile'][:,:,3:6]),
                array2image(npz['depth_tactile'][:,:,0]),
                array2image(npz['depth_tactile'][:,:,1]),
                numeric_fields(npz, idnum))

    slider.change(fn=update_frame, inputs=[slider], outputs=[rgb_static, depth_static, rgb_gripper, depth_gripper, rgb_tactile1, rgb_tactile2, depth_tactile1, depth_tactile2, text_info])
    demo.load(lambda: 1, inputs=None, outputs=slider)


demo.launch(server_name="0.0.0.0")

slider.change()
