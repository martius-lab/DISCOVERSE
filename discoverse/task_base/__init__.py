from .airbot_task_base import AirbotPlayTaskBase, recoder_airbot_play
from .mmk2_task_base import MMK2TaskBase, recoder_mmk2
from .so101_task_base import SO101TaskBase, recoder_so101
from .airbot_task_base import PyavImageEncoder

import os
import shutil
import pickle
import mediapy
from concurrent.futures import ThreadPoolExecutor


def generate_playback_script(save_dir, task_file_basename):
    """Generate record_playback.py script for visual randomization and playback"""
    playback_script = f'''"""
Record Playback Script
This script plays back a recorded trajectory and renders observations with optional visual randomization.

Usage:
    python record_playback.py --data_idx 0 [--randomize_visuals]

Or run all trajectories:
    python record_playback.py --all [--randomize_visuals]
"""

import os
import sys
import argparse
import pickle
import numpy as np
import mujoco
import mediapy
from PIL import Image
import random
import glob
from discoverse.task_base import playback_and_render

if __name__ == "__main__":
    # Import the task configuration from the parent script
    from {task_file_basename.replace(".py", "")} import cfg, SimNode

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=None, help="Data index to playback")
    parser.add_argument("--all", action="store_true", help="Process all trajectories")
    parser.add_argument("--randomize_visuals", action="store_true", help="Apply visual randomization")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to the dataset")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get list of trajectories to process
    if args.all:
        # Find all trajectory directories
        traj_dirs = sorted(glob.glob(os.path.join(script_dir, "[0-9][0-9][0-9]")))
        if not traj_dirs:
            print("No trajectory directories found")
            exit(1)
        print(f"Found {{len(traj_dirs)}} trajectories to process")
    elif args.data_idx is not None:
        traj_dirs = [os.path.join(script_dir, f"{{args.data_idx:03d}}")]
        if not os.path.exists(traj_dirs[0]):
            print(f"Data path does not exist: {{traj_dirs[0]}}")
            exit(1)
    else:
        print("Error: Must specify either --data_idx or --all")
        parser.print_help()
        exit(1)

    # Process each trajectory
    for traj_path in traj_dirs:
        traj_name = os.path.basename(traj_path)
        print(f"\\nProcessing trajectory {{traj_name}}...")

        # Initialize simulator
        cfg.headless = False  # Set to True for faster rendering without window
        sim_node = SimNode(cfg)
        playback_and_render(cfg, sim_node, traj_path, args.dataset_path, args.randomize_visuals)
        print(f"Completed {{traj_name}}")

    print(f"\\nAll done! Processed {{len(traj_dirs)}} trajectories")
'''

    playback_path = os.path.join(save_dir, "record_playback.py")
    with open(playback_path, "w") as f:
        f.write(playback_script)
    print(f"Generated playback script: {playback_path}")


def playback_and_render(cfg, sim_node, data_path, dataset_path, randomize_visuals=False):
    """Playback recorded states and render observations, saving in LeRobot format"""
    import json
    import numpy as np

    # Load recorded states and actions from episode directory (e.g., 000)
    states_file = os.path.join(data_path, "mujoco_states.pkl")
    obs_action_file = os.path.join(data_path, "obs_action.json")

    if not os.path.exists(states_file):
        print(f"No mujoco_states.pkl found in {data_path}")
        return

    if not os.path.exists(obs_action_file):
        print(f"No obs_action.json found in {data_path}")
        return

    with open(states_file, 'rb') as f:
        state_lst = pickle.load(f)

    with open(obs_action_file, 'r') as f:
        obs_action_data = json.load(f)

    # Load task description if available
    task_description = ""
    task_description_path = os.path.join(data_path, "task_description.txt")
    if os.path.exists(task_description_path):
        with open(task_description_path, mode="r") as f:
            task_description = f.read()

    print(f"Loaded {len(state_lst)} states from {data_path}")

    # Extract actions from obs_action.json
    actions = np.array(obs_action_data["act"])

    # Apply visual randomization if requested
    if randomize_visuals:
        sim_node.visual_domain_randomization()
        print("Visual randomization applied")

    # Playback states and collect observations
    obs_lst = []
    for i, state in enumerate(state_lst):
        sim_node.set_mujoco_state(state)

        # Render observations
        if sim_node.config.enable_render:
            sim_node.render()

        # Collect observations
        obs = sim_node.getObservation()
        obs_lst.append(obs)

        if i % 10 == 0:
            print(f"\rRendering: {i+1}/{len(state_lst)}", end="", flush=True)

    print("\nRendering complete")

    if hasattr(sim_node, "save_lerobot_episode"):
        sim_node.save_lerobot_episode(data_path, dataset_path, obs_lst, actions, task_description)
    else:
        print(f"Warning: save_lerobot_episode not implemented for {type(sim_node)}")


def copypy2(source_py, target_py, save_dir=None):
    """Copy task script and generate record_playback.py for visual randomization and playback"""
    shutil.copy2(source_py, target_py)

    with open(target_py, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_main_block = False
    for line in lines:
        if in_main_block:
            break
        elif line.strip().startswith('if __name__'):
            in_main_block = True
            continue
        else:
            new_lines.append(line)
    with open(target_py, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    # Generate record_playback.py script
    if save_dir is not None:
        task_basename = os.path.basename(target_py)
        generate_playback_script(save_dir, task_basename)

def encode_single_video(task):
    """编码单个视频文件"""
    try:
        data_path = task['data_path']
        output_path = task['output_path']
        fps = task['fps']

        # 加载视频帧数据
        with open(data_path, 'rb') as f:
            frames = pickle.load(f)

        # 编码视频
        mediapy.write_video(output_path, frames, fps=fps)

        # 删除临时数据文件
        os.remove(data_path)

        return True, output_path
    except Exception as e:
        return False, str(e)

def batch_encode_videos(video_tasks, max_workers=4):
    """批量编码视频，限制并发数量"""
    if not video_tasks:
        return

    print(f"开始批量编码 {len(video_tasks)} 个视频，最大并发数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_single_video, task) for task in video_tasks]

        success_count = 0
        for i, future in enumerate(futures):
            try:
                success, result = future.result()
                if success:
                    success_count += 1
                    print(f"\r编码进度: {success_count}/{len(video_tasks)}", end="", flush=True)
                else:
                    print(f"\n视频编码失败: {result}")
            except Exception as e:
                print(f"\n视频编码异常: {e}")

    print(f"\n批量编码完成，成功: {success_count}/{len(video_tasks)}")