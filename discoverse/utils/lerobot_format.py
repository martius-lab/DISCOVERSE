import argparse
import os
import json
import pickle
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor.factory import make_default_processors
from lerobot.robots.config import RobotConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_name
from lerobot.scripts.lerobot_record import DatasetRecordConfig

import mediapy
import cv2
import numpy as np
OBS_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
FPS = 20
def create_dataset(dataset_id: str, output_dir: str) -> LeRobotDataset:
    robot = SO101Follower(
        SO101FollowerConfig(
            port="/dev/tty.usbmodem5A460812391",
            use_degrees=True,
            cameras={
                "head": OpenCVCameraConfig(
                    index_or_path=0,
                    width=640,
                    height=480,
                    fps=FPS,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=1,
                    width=640,
                    height=480,
                    fps=FPS,
                ),
            },
        )
    )
    teleop_action_processor, _, robot_observation_processor = (
        make_default_processors()
    )
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=True,
        ),
    )
    sanity_check_dataset_name(dataset_id, None)
    dataset = LeRobotDataset.create(
        dataset_id,
        FPS,
        root=output_dir,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=DatasetRecordConfig.video,
        image_writer_processes=DatasetRecordConfig.num_image_writer_processes,
        image_writer_threads=DatasetRecordConfig.num_image_writer_threads_per_camera
        * len(robot.cameras),
        batch_encoding_size=DatasetRecordConfig.video_encoding_batch_size,
    )
    return dataset
def load_mp4_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video = np.array(frames)
    return video

def load_episode_data(episode_path: str):
    # Load obs_action.json
    json_path = os.path.join(episode_path, "obs_action.json")
    with open(json_path, "r") as f:
        obs_action_data = json.load(f)
    # Load mujoco_states.pkl
    pkl_path = os.path.join(episode_path, "mujoco_states.pkl")
    with open(pkl_path, "rb") as f:
        mujoco_states = pickle.load(f)
    # Load camera videos
    cam_videos = {}
    for cam_id in ["wrist", "head"]:  # assuming 2 cameras: wrist.mp4, head.mp4
        video_path = os.path.join(episode_path, f"{cam_id}.mp4")
        cam_videos[cam_id] = load_mp4_video(video_path)
    # load llm prompts if exist
    task_description = {}
    task_description_path = os.path.join(episode_path, "task_description.txt")
    if os.path.exists(task_description_path):
        with open(task_description_path, mode="r") as f:
            task_description = f.read()
    return obs_action_data, mujoco_states, cam_videos, task_description

def convert_mikel_dataset(dataset_root: str, output_dir: str) -> None:
    # dataset structure
    # each camera is a different camera in the scene
    # dataset_root (per task)
    #   <episode_id>/
    #       cam_0.mp4  # call this the head camera
    #       cam_1.mp4  # call this the wrist camera
    #       obs_action.json
    #       mujoco_states.pkl
    #       (LLM prompts files, optional)
    # create LeRobotDataset
    dataset = create_dataset(
        dataset_id="mikel_dataset/concat",
        output_dir=output_dir,
    )
    for episode_name in os.listdir(dataset_root):
        episode_path = os.path.join(dataset_root, episode_name)
        if not os.path.isdir(episode_path):
            continue
        print(f"Processing episode: {episode_name}")
        obs_action_data, mujoco_states, cam_videos, task_description = (
            load_episode_data(episode_path)
        )
        cam_videos = {
            f"{k}": v.astype(np.uint8) for k, v in cam_videos.items()
        }
        # print(obs_action_data.keys())
        # print(mujoco_states[0].keys())
        # print(cam_videos.keys())
        # covert observation_dict in np.ndarray
        observations = np.array(obs_action_data["obs"]["jq"])
        actions = np.array(obs_action_data["act"])
        # print(dataset.features)
        # build proprioceptive observation and action keys
        for t, (act, obs) in enumerate(zip(actions, observations)):
            act = {k: v for k, v in zip(ACTION_KEYS, act)}
            act = build_dataset_frame(dataset.features, act, prefix=ACTION)
            obs = {k: v for k, v in zip(OBS_KEYS, obs)}
            imgs = {k: v[t] for k, v in cam_videos.items()}
            obs = {**obs, **imgs}
            obs = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            frame = {**act, **obs, "task": task_description}
            dataset.add_frame(frame)
        dataset.save_episode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mikel's teleop dataset to LeRobot Dataset format"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Path to Mikel's dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted dataset",
    )
    args = parser.parse_args()
    convert_mikel_dataset(**vars(args))