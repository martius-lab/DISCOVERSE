import os
import json
import pickle
import numpy as np
import argparse
import shutil
import mediapy

# Configuration matching MMK2
OBS_JQ_DIM = 17
ACT_DIM = 19
QPOS_DIM = 24
QVEL_DIM = 23
FPS = 20  # Matching box_pick.py cfg.render_set["fps"] = 20
WIDTH = 640
HEIGHT = 480
DURATION = 2.0 # seconds
STEPS = int(DURATION * FPS)

def generate_dummy_data(save_dir, data_idx):
    episode_dir = os.path.join(save_dir, f"{data_idx:03d}")
    os.makedirs(episode_dir, exist_ok=True)

    # Generate dummy observations and actions
    obs_lst = []
    act_lst = []
    state_lst = []

    for i in range(STEPS):
        t = i / FPS

        # Observation
        obs = {
            "time": t,
            "jq": np.random.randn(OBS_JQ_DIM).tolist(),
            "base_position": np.random.randn(3).tolist(),
            "base_orientation": (np.random.randn(4) / (np.linalg.norm(np.random.randn(4)) + 1e-6)).tolist(),
            # "img": ... # We don't save images in json
        }
        obs_lst.append(obs)

        # Action
        act = np.random.randn(ACT_DIM).tolist()
        act_lst.append(act)

        # State
        state = {
            "time": t,
            "qpos": np.random.randn(QPOS_DIM),
            "qvel": np.random.randn(QVEL_DIM),
            "act": None,
            "ctrl": np.random.randn(ACT_DIM),
        }
        state_lst.append(state)

    # Save obs_action.json
    json_path = os.path.join(episode_dir, "obs_action.json")
    with open(json_path, "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
                "base_position": [o["base_position"] for o in obs_lst],
                "base_orientation_wxyz": [o["base_orientation"] for o in obs_lst],
            },
            "act": act_lst,
        }
        json.dump(obj, fp)

    # Save mujoco_states.pkl
    pkl_path = os.path.join(episode_dir, "mujoco_states.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(state_lst, f)

    # Create dummy video files
    # Assuming cam_0, cam_1, cam_2 based on box_pick.py cfg.obs_rgb_cam_id = [0, 1, 2]
    for cam_id in [0, 1, 2]:
        video_path = os.path.join(episode_dir, f"cam_{cam_id}.mp4")
        frames = np.random.randint(0, 255, (STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
        mediapy.write_video(video_path, frames, fps=FPS)

    print(f"Generated dummy data for episode {data_idx} in {episode_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    args = parser.parse_args()

    # Define save directory
    # We assume we are running from the project root or similar
    # We'll try to locate the data directory relative to this script or current working directory

    # If running from examples/tasks_mmk2/
    # ../../data/mmk2_pick_box_dummy

    # Let's just use a relative path "data/mmk2_pick_box_dummy" from where the script is run
    # assuming the user runs it from the project root.

    save_dir = "data/mmk2_pick_box_dummy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a dummy .mjb file
    with open(os.path.join(save_dir, "box_pick.mjb"), "wb") as f:
        f.write(b"dummy mjb")

    # Copy this script as box_pick.py to mimic the original behavior
    shutil.copy(__file__, os.path.join(save_dir, "box_pick.py"))

    for i in range(args.data_idx, args.data_idx + args.data_set_size):
        generate_dummy_data(save_dir, i)
