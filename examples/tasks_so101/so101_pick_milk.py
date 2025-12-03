import sys
import os
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
import argparse

# Add parent directory of examples to python path to import from other examples
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(current_dir)
sys.path.append(examples_dir)
sys.path.append(os.path.dirname(examples_dir)) # Add project root

from discoverse import DISCOVERSE_ASSETS_DIR, DISCOVERSE_ROOT_DIR
from discoverse.robots_env.so101_base import SO101Cfg
from discoverse.task_base import SO101TaskBase, recoder_so101
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from mocap_ik.mocap_so101_ik import SO101_IK

class SO101SimNode(SO101TaskBase):
    def domain_randomization(self):
        # No randomization for now
        pass

    def check_success(self):
        # Check if milk_0 is lifted
        try:
            milk_pos = self.mj_data.body("milk_0").xpos
        except KeyError:
            try:
                milk_pos = self.mj_data.body("milk").xpos
            except KeyError:
                return False
                
        # Table height is 0.6, so if it is > 0.65 it is lifted
        if milk_pos[2] > 0.65:
            return True
        return False

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size

    cfg = SO101Cfg()
    cfg.headless = False
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    cfg.render_set["fps"] = 30
    cfg.obs_rgb_cam_id = [0] # Record camera 0
    
    # Ensure we are using the scene with the objects
    cfg.mjcf_file_path = "mjcf/lerobot_so101/xml/so101_tabletop_manipulation_generated.xml"

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/so101_pick_milk")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SO101SimNode(cfg)
    
    # Initialize IK
    ik_xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path)
    arm_ik = SO101_IK(ik_xml_path, arm_dof=5)

    stm = SimpleStateMachine()
    stm.max_state_cnt = 5 # Define states
    
    action = np.zeros(sim_node.na)
    process_list = []

    while data_idx < data_set_size:
        # Reset
        obs = sim_node.reset()
        stm.reset()
        act_lst, obs_lst = [], []
        
        print(f"Starting task {data_idx}...")
        
        while sim_node.running:
            target_pos_local = None
            target_ori_mat_local = R.from_euler("xyz", [0, np.pi, 0]).as_matrix() # Gripper down (assuming Base is identity wrt World)
            
            # Get milk position (World Frame)
            try:
                tmat_block = get_body_tmat(sim_node.mj_data, "milk_0")
                milk_pos_world = tmat_block[:3, 3]
            except ValueError: 
                try:
                    tmat_block = get_body_tmat(sim_node.mj_data, "milk")
                    milk_pos_world = tmat_block[:3, 3]
                except ValueError:
                    milk_pos_world = np.array([0.075, 0.085, 0.65]) # Approximate pos
            
            # Get Base pose (World Frame) and compute transform to Local Frame
            tmat_base = get_site_tmat(sim_node.mj_data, "baseframe")
            T_world_base = tmat_base
            T_base_world = np.linalg.inv(T_world_base)
            
            # Convert milk pos to Base Frame
            milk_pos_local = (T_base_world @ np.append(milk_pos_world, 1.0))[:3]
            # State Machine

            if stm.trigger():
                print(f"State: {stm.state_idx}")
                if stm.state_idx == 0: # Hover
                    target_pos_local = milk_pos_local + np.array([0, 0, 0.15])
                    sim_node.tctr_gripper[:] = 1.0 
                elif stm.state_idx == 1: # Open Gripper fully
                    target_pos_local = milk_pos_local + np.array([0, 0, 0.15])
                    sim_node.tctr_gripper[:] = 1.7 
                elif stm.state_idx == 2: # Move Down
                    target_pos_local = milk_pos_local + np.array([0, 0, 0.08]) 
                    sim_node.tctr_gripper[:] = 1.7 
                elif stm.state_idx == 3: # Close Gripper
                    target_pos_local = milk_pos_local + np.array([0, 0, 0.08])
                    sim_node.tctr_gripper[:] = 0.0 
                elif stm.state_idx == 4: # Move Up
                    target_pos_local = milk_pos_local + np.array([0, 0, 0.25])
                    sim_node.tctr_gripper[:] = 0.0 
                
                # Solve IK if target_pos is set
                if target_pos_local is not None:
                    # Since base rotation is Identity, target_ori_mat (World) is same as Local
                    solution, converged = arm_ik.solve_ik(
                        target_pos=target_pos_local,
                        target_ori=target_ori_mat_local,
                        current_qpos=sim_node.sensor_arm_qpos[:5] 
                    )
                    print(f"IK Converged: {converged}, Solution: {solution}")
                    # Update target arm control
                    sim_node.tctr_arm[:] = solution
                
                stm.update()
            
            # Control loop (P-control to target)
            # Arm
            for i in range(5):
                action[i] = step_func(action[i], sim_node.tctr_arm[i], 5.0 * sim_node.delta_t)
            # Gripper 
            action[5] = step_func(action[5], sim_node.tctr_gripper[0], 5.0 * sim_node.delta_t)
            
            sim_node.updateControl(action)

            # Step
            obs, _, _, _, _ = sim_node.step(action)
            
            # Record
            if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                act_lst.append(action.tolist().copy())
                obs_lst.append(obs)
            
            # Check convergence using SimNode's checkActionDone
            if sim_node.checkActionDone():
                 stm.next()

            if sim_node.mj_data.time > 20.0:
                print("Timeout")
                break
            
            if stm.state_idx == 5:
                if sim_node.check_success():
                    print("Success!")
                    save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                    process = mp.Process(target=recoder_so101, args=(save_path, act_lst, obs_lst, cfg))
                    process.start()
                    process_list.append(process)
                    data_idx += 1
                else:
                    print("Failed to lift.")
                break
                
    for p in process_list:
        p.join()

if __name__ == "__main__":
    main()