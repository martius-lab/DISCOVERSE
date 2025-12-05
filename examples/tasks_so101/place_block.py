import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

import discoverse
from discoverse.envs import make_env
from discoverse.robots import SO101_IK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.robots_env.so101_base import SO101Cfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import SO101TaskBase, recoder_so101, copypy2
from discoverse.task_base.recording import Recording
from discoverse.task_base.airbot_task_base import PyavImageEncoder

class SimNode(SO101TaskBase):
    def __init__(self, config: SO101Cfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

    def domain_randomization(self):
        # Random block position
        self.mj_data.qpos[self.nj+0] += 2.*(np.random.random() - 0.5) * 0.05
        self.mj_data.qpos[self.nj+1] += 2.*(np.random.random() - 0.5) * 0.05

        # Random bowl position
        self.mj_data.qpos[self.nj+7+0] += 2.*(np.random.random() - 0.5) * 0.05
        self.mj_data.qpos[self.nj+7+1] += 2.*(np.random.random() - 0.5) * 0.05


    def check_success(self):
        tmat_block = get_body_tmat(self.mj_data, "block_green")
        tmat_bowl = get_body_tmat(self.mj_data, "bowl_pink")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_block[0, 3] - tmat_bowl[0, 3], tmat_block[1, 3] - tmat_bowl[1, 3]) < 0.02


save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", "so101_"+os.path.splitext(os.path.basename(__file__))[0])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cfg = SO101Cfg()
robot_name = "so101"
task_name = "place_block"
cfg.mjcf_file_path = f"{save_dir}/{task_name}.xml"


env = make_env(robot_name, task_name)
env.export_xml(os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path))

cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = False

if __name__ == "__main__":

    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=10, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--save_segment", action="store_true", help="save segment videos")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    parser.add_argument("--render", action="store_true", help="Render observations after simulation (runs record_playback.py)")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    sim_node = SimNode(cfg)
    if data_idx == 0:
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)), save_dir)

    arm_ik = SO101_IK()

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()

    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 9
    max_time = 10.0  # seconds

    action = np.zeros(6)
    recorder = Recording(save_dir, cfg, start_idx=data_idx)

    move_speed = 0.8
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            recorder.reset()
            save_path = os.path.join(save_dir, "{:03d}".format(recorder.data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger():
                print(f"State: {stm.state_idx}")
                if stm.state_idx == 0: # Move to above the block
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])
                    sim_node.target_control[5] = 1.0
                elif stm.state_idx == 1: # Move to the block
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.008 * tmat_jujube[:3, 2] + 0.01 * tmat_jujube[:3, 0] # x
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])
                elif stm.state_idx == 2: # Grasp the block
                    sim_node.target_control[5] = 0.0
                elif stm.state_idx == 3: # Stabilize the block
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # Lift the block
                    tmat_tgt_local[2,3] += 0.07
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])
                elif stm.state_idx == 5: # Move the block to above the bowl
                    tmat_plate = get_body_tmat(sim_node.mj_data, "bowl_pink")
                    tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.0, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])
                elif stm.state_idx == 6: # Lower the height, place the block on the bowl
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])
                elif stm.state_idx == 7: # Release the block
                    sim_node.target_control[5] = 1.0
                    print("Releasing the block")
                    print("Set the gripper to:", sim_node.target_control[5])
                elif stm.state_idx == 8: # Lift height
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:5] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:5])

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)

        obs, _, _, _, _ = sim_node.step(action)

        if len(recorder.obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            recorder.add(action.tolist().copy(), obs, sim_node.get_mujoco_state())

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recorder.record_episode(recoder_so101)
                print("\r{:4}/{:4} ".format(recorder.data_idx, data_set_size), end="")
                if recorder.data_idx >= data_set_size:
                    break
            else:
                print(f"{recorder.data_idx} Failed")

            obs = sim_node.reset()

    recorder.finish_recording()

    # If --render flag is provided, run record_playback.py to render observations
    if args.render:
        recorder.export_to_lerobot()
