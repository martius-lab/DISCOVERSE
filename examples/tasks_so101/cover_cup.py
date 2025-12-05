import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse

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
        self.tmat_tgt_local = None

    def domain_randomization(self):
        # Randomize object positions if needed
        pass

    def check_success(self):
        tmat_lid = get_body_tmat(self.mj_data, "cup_lid")
        tmat_cup = get_body_tmat(self.mj_data, "coffeecup_white")
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")
        return (abs(tmat_cup[2, 2]) > 0.99) and \
            np.hypot(tmat_plate[0, 3] - tmat_cup[0, 3], tmat_plate[1, 3] - tmat_cup[1, 3]) < 0.02 and \
            np.hypot(tmat_lid[0, 3] - tmat_cup[0, 3], tmat_lid[1, 3] - tmat_cup[1, 3]) < 0.02

    def move_to_pre_grasp_cup(self, arm_ik, tmat_armbase_2_world, trmat_cup_quat):
        tmat_coffee = get_body_tmat(self.mj_data, "coffeecup_white").copy()
        tmat_coffee[:3, 3] = tmat_coffee[:3, 3] + 0.1 * tmat_coffee[:3, 1] + 0.1 * tmat_coffee[:3, 2]
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_coffee
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])
        self.target_control[5] = 1.0

    def move_to_grasp_cup(self, arm_ik, tmat_armbase_2_world, trmat_cup_quat):
        tmat_coffee = get_body_tmat(self.mj_data, "coffeecup_white").copy()
        # Position at handle
        tmat_coffee[:3, 3] = tmat_coffee[:3, 3] + 0.06 * tmat_coffee[:3, 1] + 0.05 * tmat_coffee[:3, 2]
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_coffee
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def grasp_object(self):
        self.target_control[5] = 0.0

    def stabilize_grasp(self):
        self.delay_cnt = int(0.25/self.delta_t)

    def lift_cup(self, arm_ik, trmat_cup_quat):
        self.tmat_tgt_local[2,3] += 0.15
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def move_cup_to_above_plate(self, arm_ik, tmat_armbase_2_world, trmat_cup_quat):
        tmat_plate = get_body_tmat(self.mj_data, "plate_white").copy()
        tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.06, 0.0, 0.13])
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def move_cup_to_above_plate_lower(self, arm_ik, tmat_armbase_2_world, trmat_cup_quat):
        tmat_plate = get_body_tmat(self.mj_data, "plate_white").copy()
        tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.06, 0.0, 0.08])
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def place_cup_on_plate(self, arm_ik, trmat_cup_quat):
        self.tmat_tgt_local[2,3] -= 0.02
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def release_object(self):
        self.target_control[5] = 1.0

    def retract_arm(self, arm_ik, trmat_cup_quat):
        self.tmat_tgt_local[2,3] += 0.08
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_cup_quat, self.mj_data.qpos[:5])

    def move_to_above_lid(self, arm_ik, tmat_armbase_2_world, trmat_lid_quat):
        tmat_lid = get_body_tmat(self.mj_data, "cup_lid").copy()
        tmat_lid[:3, 3] = tmat_lid[:3, 3] + 0.1 * tmat_lid[:3, 2]
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_lid
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

    def lower_to_lid(self, arm_ik, trmat_lid_quat):
        self.tmat_tgt_local[2,3] -= 0.04
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

    def lift_lid(self, arm_ik, trmat_lid_quat):
        self.tmat_tgt_local[2,3] += 0.08
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

    def move_lid_to_above_cup(self, arm_ik, tmat_armbase_2_world, trmat_lid_quat):
        tmat_cup = get_body_tmat(self.mj_data, "coffeecup_white").copy()
        tmat_cup[:3,3] = tmat_cup[:3, 3] + np.array([0.0, 0.0, 0.16])
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_cup
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

    def place_lid_on_cup(self, arm_ik, trmat_lid_quat):
        self.tmat_tgt_local[2,3] -= 0.02
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

    def final_retract(self, arm_ik, trmat_lid_quat):
        self.tmat_tgt_local[2,3] += 0.05
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat_lid_quat, self.mj_data.qpos[:5])

cfg = SO101Cfg()
cfg.gs_model_dict["background"]      = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]        = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]        = "hinge/drawer_2.ply"
cfg.gs_model_dict["coffeecup_white"] = "object/teacup.ply"
cfg.gs_model_dict["plate_white"]     = "object/plate_white.ply"
cfg.gs_model_dict["wood"]            = "object/wood.ply"
cfg.gs_model_dict["cup_lid"]         = "object/teacup_lid.ply"
cfg.init_qpos[:] = [0, 0, 0, 0, 0, 0.0]

task_name = "cover_cup"
robot_name = "so101"
cfg.mjcf_file_path = f"mjcf/tmp/{robot_name}_{task_name}.xml"
env = make_env(robot_name, task_name)
env.export_xml(os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path))

cfg.obj_list     = ["drawer_1", "drawer_2", "coffeecup_white", "plate_white", "cup_lid"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id   = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":

    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--save_segment", action="store_true", help="save segment videos")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    arm_ik = SO101_IK()

    trmat_cup = Rotation.from_euler("xyz", [0, np.pi, 0], degrees=False).as_matrix()
    trmat_lid = Rotation.from_euler("xyz", [0, np.pi, 0], degrees=False).as_matrix()

    # Pass rotation matrix directly or quaternion if preferred. SO101_IK supports both.
    # Converting to quaternion for consistency with previous code
    trmat_cup_quat = Rotation.from_matrix(trmat_cup).as_quat()[[3, 0, 1, 2]]
    trmat_lid_quat = Rotation.from_matrix(trmat_lid).as_quat()[[3, 0, 1, 2]]

    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 18
    max_time = 20.0 #s

    action = np.zeros(6)
    recorder = Recording(save_dir, cfg, start_idx=data_idx)

    move_speed = 0.8
    sim_node.reset() # This should be called after sim_node is initialized

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
                if stm.state_idx == 0: # Approach cup (pre-grasp position)
                    sim_node.move_to_pre_grasp_cup(arm_ik, tmat_armbase_2_world, trmat_cup_quat)
                elif stm.state_idx == 1: # Move to handle grasp position
                    sim_node.move_to_grasp_cup(arm_ik, tmat_armbase_2_world, trmat_cup_quat)
                elif stm.state_idx == 2: # Close gripper
                    sim_node.grasp_object()
                elif stm.state_idx == 3: # Wait for grasp
                    sim_node.stabilize_grasp()
                elif stm.state_idx == 4: # Lift cup
                    sim_node.lift_cup(arm_ik, trmat_cup_quat)
                elif stm.state_idx == 5: # Move above plate (high)
                    sim_node.move_cup_to_above_plate(arm_ik, tmat_armbase_2_world, trmat_cup_quat)
                elif stm.state_idx == 6: # Move above plate (medium)
                    sim_node.move_cup_to_above_plate_lower(arm_ik, tmat_armbase_2_world, trmat_cup_quat)
                elif stm.state_idx == 7: # Lower cup onto plate
                    sim_node.place_cup_on_plate(arm_ik, trmat_cup_quat)
                elif stm.state_idx == 8: # Release cup
                    sim_node.release_object()
                elif stm.state_idx == 9: # Retract upward
                    sim_node.retract_arm(arm_ik, trmat_cup_quat)
                elif stm.state_idx == 10: # Move above lid
                    sim_node.move_to_above_lid(arm_ik, tmat_armbase_2_world, trmat_lid_quat)
                elif stm.state_idx == 11: # Lower to lid
                    sim_node.lower_to_lid(arm_ik, trmat_lid_quat)
                elif stm.state_idx == 12: # Grasp lid
                    sim_node.grasp_object()
                elif stm.state_idx == 13: # Wait for lid grasp
                    sim_node.stabilize_grasp()
                elif stm.state_idx == 14: # Lift lid
                    sim_node.lift_lid(arm_ik, trmat_lid_quat)
                elif stm.state_idx == 15: # Move lid above cup
                    sim_node.move_lid_to_above_cup(arm_ik, tmat_armbase_2_world, trmat_lid_quat)
                elif stm.state_idx == 16: # Lower lid onto cup
                    sim_node.place_lid_on_cup(arm_ik, trmat_lid_quat)
                elif stm.state_idx == 17: # Release lid
                    sim_node.release_object()
                elif stm.state_idx == 18: # Final retract
                    sim_node.final_retract(arm_ik, trmat_lid_quat)

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

        # Smooth gripper movement: iterate over all nj joints including gripper (index 5)
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
                recorder.record_episode(recoder_so101)
                break
            sim_node.reset()

    recorder.finish_recording()

    # If --render flag is provided, run record_playback.py to render observations
    if hasattr(args, "render") and args.render:
        recorder.export_to_lerobot()