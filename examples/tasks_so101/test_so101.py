import sys
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# Add parent directory of examples to python path to import from other examples
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(current_dir)
sys.path.append(examples_dir)
sys.path.append(os.path.dirname(examples_dir)) 

from mocap_ik.mocap_so101_ik import SO101_IK
from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.robots_env.so101_base import SO101Cfg, SO101Base
from discoverse.utils import get_body_tmat, step_func

def test_so101():
    # 1. Initialize Environment
    cfg = SO101Cfg()
    cfg.headless = False
    sim_node = SO101Base(cfg)
    
    # 2. Initialize IK Solver
    # Use the same MJCF for IK to include scene obstacles if needed, 
    # or just the robot if we want it faster/simpler. 
    # Here we use the full scene config from the env.
    ik_xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path)
    arm_ik = SO101_IK(ik_xml_path, arm_dof=5)
    
    # 3. Simulation Loop
    sim_node.reset()
    action = np.zeros(sim_node.na)
    
    target_block_name = "milk"
    
    print(f"Moving to {target_block_name}...")
    target_pos_printed = False
    
    while sim_node.running:
        # Get target position (block)
        try:
            tmat_block = get_body_tmat(sim_node.mj_data, target_block_name)
        except ValueError:
            print(f"Body {target_block_name} not found!")
            break
            
        if not target_pos_printed:
            # Debug prints
            cube_body_id = mujoco.mj_name2id(sim_node.mj_model, mujoco.mjtObj.mjOBJ_BODY, target_block_name)
            print(f"Cube Body ID: {cube_body_id}")
            if cube_body_id >= 0:
                 print(f"Cube Model Pos: {sim_node.mj_model.body_pos[cube_body_id]}")
                 print(f"Cube Data XPos: {sim_node.mj_data.xpos[cube_body_id]}")
            
            target_pos = tmat_block[:3, 3] + np.array([0, 0, 0.15]) # Hover 15cm above
            print(f"Target Pos: {target_pos}")
            target_pos_printed = True
        
        # Define target orientation (gripper down)
        # SO101 default might vary, assuming Z-down approach
        target_ori_mat = R.from_euler("xyz", [0, np.pi, 0]).as_matrix()
        
        # Current joint positions
        qpos = sim_node.sensor_arm_qpos
        
        # Solve IK
        solution, converged = arm_ik.solve_ik(
            target_pos=target_pos,
            target_ori=target_ori_mat,
            current_qpos=qpos
        )
        
        # Apply solution regardless of convergence for debug
        for i in range(len(solution)):
            action[i] = step_func(action[i], solution[i], 5.0 * sim_node.delta_t)
            
        if not converged and int(sim_node.mj_data.time * 10) % 10 == 0:
             # Print occasionally
             pass 
             # print("IK attempting (not converged)...")
            
        # Step simulation
        sim_node.step(action)

        if sim_node.mj_data.time > 10.0:
            print("Timeout")
            break

if __name__ == "__main__":
    test_so101()
