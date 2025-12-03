"""Test if we can do IK for SO-101 arm. We load the base XML file from so101_reach.xml. It converges to a target position and orientation."""

import os
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.utils import get_site_tmat
from discoverse import DISCOVERSE_ASSETS_DIR
from typing import Tuple, Optional


class SO101_IK:
    def __init__(self, mjcf_path, arm_dof):
        self.arm_dof = arm_dof
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.configuration = mink.Configuration(self.mj_model)
        self.end_effector_task = mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.mj_model, cost=1e-2)
        self.mink_tasks = [self.end_effector_task, self.posture_task]

        self.posture_task.set_target_from_configuration(self.configuration)

        self.solver = "quadprog"
        self.pos_threshold = 5e-3
        self.ori_threshold = 1e-2
        self.max_iters = 200

    def converge_ik(self, dt=0.0):
        dt = dt or self.mj_model.opt.timestep * 2
        for _ in range(self.max_iters):
            vel = mink.solve_ik(self.configuration, self.mink_tasks, dt, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, dt)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                return True
        return False
    
    def fk(self, current_qpos):
        # Update current configuration
        tmp_q = self.configuration.data.qpos.copy()
        # Only update the arm joints, keep others (gripper) as is
        tmp_q[:len(current_qpos)] = current_qpos[:]
        self.configuration.update(tmp_q)
        tmat_base = get_site_tmat(self.configuration.data, "baseframe")
        tmat_endpoint = get_site_tmat(self.configuration.data, "gripperframe")
        tmat_local = np.linalg.inv(tmat_base) @ tmat_endpoint
        target_position = tmat_local[:3,3]
        target_quat_wxyz = Rotation.from_matrix(tmat_local[:3,:3]).as_quat()[[3,0,1,2]]
        return target_position, target_quat_wxyz

    def solve_ik(self, 
                 target_pos: np.ndarray, 
                 target_ori: np.ndarray, 
                 current_qpos: np.ndarray,
                 reference_qpos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve IK
        
        Args:
            target_pos: Target position [x, y, z]
            target_ori: Target orientation matrix (3x3) or quaternion [qw, qx, qy, qz]
            current_qpos: Current joint positions
            reference_qpos: Reference joint positions (for posture task)
            
        Returns:
            Tuple[Joint positions, Converged boolean]
        """
        print("Solving IK...")
        # Update current configuration
        tmp_q = self.configuration.data.qpos.copy()
        tmp_q[:len(current_qpos)] = current_qpos[:]
        self.configuration.update(tmp_q)

        # Handle target orientation
        if target_ori.shape == (4,):
            # Quaternion to rotation matrix
            target_rot_matrix = Rotation.from_quat(target_ori[[1, 2, 3, 0]]).as_matrix()
        elif target_ori.shape == (3, 3):
            target_rot_matrix = target_ori
        else:
            raise ValueError(f"Invalid target orientation shape: {target_ori.shape}")
        
        # Build target transformation matrix
        T_target = np.eye(4)
        T_target[:3, :3] = target_rot_matrix
        T_target[:3, 3] = target_pos
        tmat_base = get_site_tmat(self.configuration.data, "baseframe")
        
        # Set target
        target_SE3 = mink.SE3.from_matrix(tmat_base @ T_target)
        self.end_effector_task.set_target(target_SE3)
        
        # If reference position provided, update posture task
        if reference_qpos is not None:
            temp_config = mink.Configuration(self.mj_model)
            temp_config.update(reference_qpos)
            self.posture_task.set_target_from_configuration(temp_config)
        
        converged = self.converge_ik(0.005)
        solution = self.configuration.data.qpos[:self.arm_dof]

        return solution, converged

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    # Correct path to SO-101 XML
    mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/lerobot_so101/xml/so101_reach.xml")
    print(f"Loading model from: {mjcf_path}")
    
    # Initialize IK solver
    mik = SO101_IK(mjcf_path, arm_dof=5)

    # Get current pose (FK)
    current_qpos = np.zeros(5)
    curr_pos, curr_quat = mik.fk(current_qpos)
    print(f"Current Position: {curr_pos}")
    print(f"Current Orientation (wxyz): {curr_quat}")

    # Define a target (e.g., slightly moved from current)
    target_pos = curr_pos + np.array([0.05, 0.0, -0.05])
    target_ori = curr_quat # Keep same orientation
    
    print(f"Target Position: {target_pos}")

    # Solve IK
    solution, converged = mik.solve_ik(
        target_pos = target_pos,
        target_ori = target_ori,
        current_qpos = current_qpos
    )

    print(f"Converged: {converged}")
    print(f"Solution: {solution}")
    
    # Verify solution with FK
    sol_pos, sol_quat = mik.fk(solution)
    print(f"Achieved Position: {sol_pos}")
    print(f"Achieved Orientation (wxyz): {sol_quat}")
    
    pos_err = np.linalg.norm(sol_pos - target_pos)
    print(f"Position Error: {pos_err:.6f}")
