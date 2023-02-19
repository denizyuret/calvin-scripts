# python visualize_calvin_npz.py /datasets/calvin/D/validation/episode_0000000.npz
# visualize a single frame, modified from the original visualize_calvin.py

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument("path", type=str, help="Path to dir containing scene_info.npy")
    parser.add_argument("-d", "--data", nargs="*", default=["rgb_static"], help="Data to visualize")
    args = parser.parse_args()

    if not Path(args.path).is_file():
        print(f"Path {args.path} is either not a file, or does not exist.")
        exit()

    while True:
        t = np.load(f"{args.path}", allow_pickle=True)

        for d in args.data:
            if d not in t:
                print(f"Data {d} cannot be found in transition")
                continue

            cv2.imshow(d, t[d][:, :, ::-1])

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == 83:  # Right arrow
            idx = (idx + 1) % len(indices)
        elif key == 81:  # Left arrow
            idx = (len(indices) + idx - 1) % len(indices)
        else:
            print(f'Unrecognized keycode "{key}"')
