"""Test if we can do IK for SO-101 arm. We load the base XML file from so101_reach.xml. It converges to a target position and orientation."""

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.utils import get_site_tmat
from typing import Tuple, Optional

so101_xml = '''
<mujoco model="so101-ik">
<compiler angle="radian" autolimits="true"/>
<worldbody>
    <!-- SO101 Robot -->
    <body name="base" pos="0 0 0" quat="1 0 0 0">
      <inertial pos="0.0137179 -5.19711e-05 0.0334843" mass="0.147" fullinertia="0.000114686 0.000136117 0.000130364 -4.59787e-07 4.97151e-06 9.75275e-08"/>
      <site group="3" name="baseframe" pos="8.67362e-19 9.55596e-18 3.46945e-18" quat="1 -8.17396e-19 3.78392e-17 2.22045e-16"/>
      <body name="shoulder" pos="0.0388353 -8.97657e-09 0.0624" quat="3.56167e-16 1.22818e-15 -1 -4.14635e-16">
        <joint axis="0 0 1" name="shoulder_pan" type="hinge" range="-1.9198621771937616 1.9198621771937634" />
        <inertial pos="-0.0307604 -1.66727e-05 -0.0252713" mass="0.100006" fullinertia="8.3759e-05 8.10403e-05 2.39783e-05 7.55525e-08 -1.16342e-06 1.54663e-07"/>
        <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="0.5 -0.5 -0.5 -0.5">
          <joint axis="0 0 1" name="shoulder_lift" type="hinge" range="-1.7453292519943224 1.7453292519943366" />
          <inertial pos="-0.0898471 -0.00838224 0.0184089" mass="0.103" fullinertia="4.08002e-05 0.000147318 0.000142487 -1.97819e-05 -4.03016e-08 8.97326e-09"/>
          <body name="lower_arm" pos="-0.11257 -0.028 1.73763e-16" quat="0.707107 -5.98613e-17 -2.58051e-17 0.707107">
            <joint axis="0 0 1" name="elbow_flex" type="hinge" range="-1.69 1.69" />
            <inertial pos="-0.0980701 0.00324376 0.0182831" mass="0.104" fullinertia="2.87438e-05 0.000159844 0.00014529 7.41152e-06 1.26409e-06 -4.90188e-08"/>
            <body name="wrist" pos="-0.1349 0.0052 3.62355e-17" quat="0.707107 9.58722e-16 -7.51313e-16 -0.707107">
              <joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.6580628494556928 1.6580627293335335" />
              <inertial pos="-0.000103312 -0.0386143 0.0281156" mass="0.079" fullinertia="3.68263e-05 2.5391e-05 2.1e-05 1.7893e-08 -5.28128e-08 3.6412e-06"/>
              <body name="gripper" pos="5.55112e-17 -0.0611 0.0181" quat="0.0172091 -0.0172091 0.706897 0.706897">
                <joint axis="0 0 1" name="wrist_roll" type="hinge" range="-2.7438472969992493 2.841206309382605" />
                <site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/>
                <inertial pos="0.000213627 0.000245138 -0.025187" mass="0.087" fullinertia="2.75087e-05 4.33657e-05 3.45059e-05 -3.35241e-07 -5.7352e-06 -5.17847e-08"/>
                <body name="moving_jaw_so101_v1" pos="0.0202 0.0188 -0.0234" quat="0.707107 0.707107 -1.85362e-08 1.85362e-08">
                  <joint axis="0 0 1" name="gripper" type="hinge" range="-0.17453297762778586 1.7453291995659765" />
                  <inertial pos="-0.00157495 -0.0300244 0.0192755" mass="0.012" fullinertia="6.61427e-06 1.89032e-06 5.28738e-06 -3.19807e-07 -5.90717e-09 -1.09945e-07"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" forcerange="-3.35 3.35" ctrlrange="-1.91986 1.91986"/>
    <position name="shoulder_lift" joint="shoulder_lift" forcerange="-3.35 3.35" ctrlrange="-1.74533 1.74533"/>
    <position name="elbow_flex" joint="elbow_flex" forcerange="-3.35 3.35" ctrlrange="-1.69 1.69"/>
    <position name="wrist_flex" joint="wrist_flex" forcerange="-3.35 3.35" ctrlrange="-1.65806 1.65806"/>
    <position name="wrist_roll" joint="wrist_roll" forcerange="-3.35 3.35" ctrlrange="-2.74385 2.84121"/>
    <position name="gripper" joint="gripper" forcerange="-3.35 3.35" ctrlrange="-0.17453 1.74533"/>
  </actuator>

</mujoco>
'''
class SO101_IK:
    def __init__(self):
        self.arm_dof = 5
        self.mj_model = mujoco.MjModel.from_xml_string(so101_xml)
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

        return solution #, converged
    
    def properIK(self, 
                 target_pos: np.ndarray, 
                 target_ori: np.ndarray, 
                 current_qpos: np.ndarray,
                 reference_qpos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        return self.solve_ik(target_pos, 
                 target_ori, 
                 current_qpos,
                 reference_qpos)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    
    # Initialize IK solver
    mik = SO101_IK()

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
