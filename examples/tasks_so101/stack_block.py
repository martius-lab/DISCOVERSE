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

class SimNode(SO101TaskBase):
    def __init__(self, config: SO101Cfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())
        self.tmat_tgt_local = None

    def domain_randomization(self):
        # Randomize block_green position
        flag_position = False
        while not flag_position:

            self.object_pose("block_red")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])
            self.object_pose("block_green")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])
            self.object_pose("block_blue")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])

            position_list = np.array([
                self.object_pose("block_red")[:2],
                self.object_pose("block_green")[:2],
                self.object_pose("block_blue")[:2]])

            flag_position = self.check_position(position_list, 0.03)

        # Randomize eye side view
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        tmat_block_green = get_body_tmat(self.mj_data, "block_green")
        tmat_block_blue = get_body_tmat(self.mj_data, "block_blue")
        tmat_block_red = get_body_tmat(self.mj_data, "block_red")
        return (abs(tmat_block_blue[2, 2]) > 0.99) and \
            np.hypot(tmat_block_green[0, 3] - tmat_block_blue[0, 3], tmat_block_green[1, 3] - tmat_block_blue[1, 3]) < 0.02 and \
            np.hypot(tmat_block_red[0, 3] - tmat_block_blue[0, 3], tmat_block_red[1, 3] - tmat_block_blue[1, 3]) < 0.02

    def check_position(self, position_list, tolerance):
        for i in range(len(position_list)-1):
            if i < len(position_list) - 1:
                res = np.linalg.norm(position_list[i] - position_list[i+1:], axis=1)
            else:
                res = np.linalg.norm(position_list[i] - position_list[i+1])
            if np.any(res < tolerance):
                return False
        return True

    def move_to_above_block(self, arm_ik, tmat_armbase_2_world, trmat, block_name):
        tmat_jujube = get_body_tmat(self.mj_data, block_name)
        tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])
        self.target_control[5] = 1.0

    def move_block_to_above_target(self, arm_ik, tmat_armbase_2_world, trmat, target_name, offset_z):
        tmat_target = get_body_tmat(self.mj_data, target_name)
        tmat_target[:3,3] = tmat_target[:3, 3] + np.array([0.0, 0.0, offset_z]) + 0.008 * tmat_target[:3, 2] + 0.01 * tmat_target[:3, 0]
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_target
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])

    def move_to_block(self, arm_ik, tmat_armbase_2_world, trmat, block_name):
        tmat_jujube = get_body_tmat(self.mj_data, block_name)
        tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.008 * tmat_jujube[:3, 2] + 0.01 * tmat_jujube[:3, 0] # x
        self.tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])

    def lower_to_other_box(self, arm_ik, trmat, lower_height):
        self.tmat_tgt_local[2,3] -= lower_height
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])

    def grasp_block(self):
        self.target_control[5] = 0.1

    def stabilize_block(self):
        self.delay_cnt = int(0.35/self.delta_t)

    def lift_block(self, arm_ik, trmat, lift_height=0.07):
        self.tmat_tgt_local[2,3] += lift_height
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])



    def release_block(self):
        self.target_control[5] = 1.0

    def lift_height(self, arm_ik, trmat, lift_h=0.05):
        self.tmat_tgt_local[2,3] += lift_h
        self.target_control[:5] = arm_ik.properIK(self.tmat_tgt_local[:3,3], trmat, self.mj_data.qpos[:5])

cfg = SO101Cfg()
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"
cfg.init_qpos[:] = [0, 0, 0, 0, 0, 0.0]

robot_name = "so101"
task_name = "stack_block"
cfg.mjcf_file_path = f"mjcf/tmp/{robot_name}_{task_name}.xml"
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
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":

    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=100, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
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

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 18
    max_time = 16.0 # seconds

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

        try:
            if stm.trigger():
                if stm.state_idx == 0: # Reach above the block
                    sim_node.move_to_above_block(arm_ik, tmat_armbase_2_world, trmat, "block_green")
                elif stm.state_idx == 1: # Reach the block
                    sim_node.move_to_block(arm_ik, tmat_armbase_2_world, trmat, "block_green")
                elif stm.state_idx == 2: # Grasp the block
                    sim_node.grasp_block()
                elif stm.state_idx == 3: # Stabilize grasp on the block
                    sim_node.stabilize_block()
                elif stm.state_idx == 4: # Lift the block
                    sim_node.lift_block(arm_ik, trmat, lift_height=0.07)
                elif stm.state_idx == 5: # Move the block above the blue block
                    sim_node.move_block_to_above_target(arm_ik, tmat_armbase_2_world, trmat, "block_blue", offset_z=0.055)
                elif stm.state_idx == 6: # Lower height, place block on the blue block
                    sim_node.lower_to_other_box(arm_ik, trmat, lower_height=0.04)
                elif stm.state_idx == 7: # Release the block
                    sim_node.release_block()
                elif stm.state_idx == 8: # Lift height
                    sim_node.lift_height(arm_ik, trmat, lift_h=0.05)
                elif stm.state_idx == 9: # Reach above red block
                    sim_node.move_to_above_block(arm_ik, tmat_armbase_2_world, trmat, "block_red")
                elif stm.state_idx == 10: # Reach red block
                    sim_node.move_to_block(arm_ik, tmat_armbase_2_world, trmat, "block_red")
                elif stm.state_idx == 11: # Grasp red block
                    sim_node.grasp_block()
                elif stm.state_idx == 12: # Stabilize grasp on red block
                    sim_node.stabilize_block()
                elif stm.state_idx == 13: # Lift red block
                    sim_node.lift_block(arm_ik, trmat, lift_height=0.07)
                elif stm.state_idx == 14: # Place red block on block_green
                    sim_node.move_block_to_above_target(arm_ik, tmat_armbase_2_world, trmat, "block_green", offset_z=0.055)
                elif stm.state_idx == 15: # Lower height
                    sim_node.lower_to_other_box(arm_ik, trmat, lower_height=0.04)
                elif stm.state_idx == 16: # Release block_red
                    sim_node.release_block()
                elif stm.state_idx == 17: # Lift height
                    sim_node.lift_height(arm_ik, trmat, lift_h=0.05)

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
            print("Error: ", ve)
            print("Current Errot idx: ", stm.state_idx)
            stm.state_idx = stm.max_state_cnt + 1
            # sim_node.reset()

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
                recorder.record_episode(recoder_so101, success=False)

            sim_node.reset()

    recorder.finish_recording()

    # If --render flag is provided, run record_playback.py to render observations
    if hasattr(args, "render") and args.render:
        recorder.export_to_lerobot()
