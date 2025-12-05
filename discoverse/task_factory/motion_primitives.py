"""
Motion Primitives for Task Factory

This module defines reusable motion primitives that can be configured via YAML.
Each primitive can use get_body_tmat to trace object poses with optional offsets and rotations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.utils import get_body_tmat, get_site_tmat
from discoverse import DISCOVERSE_ROOT_DIR
from pathlib import Path
import os
import yaml

class MotionPrimitive:
    """Base class for motion primitives"""
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.object_config = None
        if 'object' in params.keys():
            object_config_path = Path(os.path.join(DISCOVERSE_ROOT_DIR, "models/meshes/library_objects", self.params['object'], self.params['object'] + ".yaml"))
            if object_config_path.is_file():
                with open(object_config_path, 'r') as object_config_yaml:
                    self.object_config = yaml.safe_load(object_config_yaml)
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        """Execute the motion primitive and return target control values"""
        raise NotImplementedError
    
    def get_target_tmat(self, sim_node, body_name, offset=None, rotation=None):
        """
        Get target transformation matrix for a body with optional offset and rotation.
        
        Args:
            sim_node: The simulation node
            body_name: Name of the body to track
            offset: [x, y, z] offset in body frame or world frame depending on context
            rotation: Rotation specification as [axis, angle] or euler angles [rx, ry, rz]
        
        Returns:
            4x4 transformation matrix
        """
        tmat = get_body_tmat(sim_node.mj_data, body_name)
        
        if offset is not None:
            offset = np.array(offset)
            if len(offset) == 3:
                tmat[:3, 3] += offset
        
        if rotation is not None:
            if isinstance(rotation, dict):
                if 'euler' in rotation:
                    rot = Rotation.from_euler(rotation.get('seq', 'xyz'), 
                                             rotation['euler'], 
                                             degrees=rotation.get('degrees', False))
                elif 'matrix' in rotation:
                    rot = Rotation.from_matrix(rotation['matrix'])
                else:
                    rot = Rotation.identity()
                tmat[:3, :3] = rot.as_matrix() @ tmat[:3, :3]
        
        return tmat


class MoveToObjectPrimitive(MotionPrimitive):
    """
    Move end-effector to an object position with optional offset and rotation.
    
    YAML params:
        object: str - Object body name
        offset: [x, y, z] - Optional offset in meters
        rotation: dict - Optional rotation specification
        gripper: float - Gripper position (for last joint, typically gripper)
    
    Note: Assumes arm joints are robot.nj-1 and last joint is gripper.
    For airbot_play: 6 arm joints + 1 gripper = 7 total
    For so101: 5 arm joints + 1 gripper = 6 total
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        object_name = self.params['object']
        offset = self.params.get('offset', None)
        rotation = self.params.get('rotation', None)
        gripper = self.params.get('gripper', None)
        
        tmat_target = self.get_target_tmat(sim_node, object_name, offset, rotation)
        tmat_target_local = tmat_armbase_2_world @ tmat_target
        
        target_control = sim_node.target_control.copy()
        # Use robot's nj-1 for arm joints (assumes last joint is gripper)
        arm_joints = sim_node.nj - 1
        target_control[:arm_joints] = arm_ik.properIK(
            tmat_target_local[:3, 3], 
            tmat_target_local[:3, :3], 
            sim_node.mj_data.qpos[:arm_joints]
        )
        
        if gripper is not None:
            target_control[sim_node.nj - 1] = gripper
        
        return target_control


class MoveToPositionPrimitive(MotionPrimitive):
    """
    Move end-effector to an absolute or relative position.
    
    YAML params:
        position: [x, y, z] - Target position
        rotation: dict - Optional rotation specification
        relative: bool - If true, position is relative to current
        gripper: float - Gripper position
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        position = np.array(self.params['position'])
        rotation = self.params.get('rotation', None)
        relative = self.params.get('relative', False)
        gripper = self.params.get('gripper', None)
        
        if relative:
            current_tmat = get_site_tmat(sim_node.mj_data, "endpoint") 
            position = current_tmat[:3, 3] + position
        
        tmat_target = np.eye(4)
        tmat_target[:3, 3] = position
        
        if rotation:
            if 'euler' in rotation:
                rot = Rotation.from_euler(rotation.get('seq', 'xyz'), 
                                         rotation['euler'], 
                                         degrees=rotation.get('degrees', False))
                tmat_target[:3, :3] = rot.as_matrix()
        
        tmat_target_local = tmat_armbase_2_world @ tmat_target
        
        target_control = sim_node.target_control.copy()
        arm_joints = sim_node.nj - 1
        target_control[:arm_joints] = arm_ik.properIK(
            tmat_target_local[:3, 3], 
            tmat_target_local[:3, :3], 
            sim_node.mj_data.qpos[:arm_joints]
        )
        
        if gripper is not None:
            target_control[sim_node.nj - 1] = gripper
        
        return target_control


class GraspPrimitive(MotionPrimitive):
    """
    Close gripper to grasp an object.
    
    YAML params:
        position: float - Gripper close position (default: 0.0)
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        # How about I modify this primitive to sample all possible graspable positions,
        # measure which one is the closest to the current pose of the gripper, and execute
        # pretty much the move to pose primitive and then close the gripper?
        # Let's try to do this.
        object_name = self.params['object']
        tmat_object = self.get_target_tmat(sim_node, object_name)
        gripper_jaw_length = 8.5e-2 # m
        # Query potential grasp poses from the config
        grasping_poses = self.object_config['grasping_poses']
        # Just choose the first possible feasible grasping pose for now.
        R_grasp_pose_claw = Rotation.from_euler('XYZ', np.array([0.0, 90.0, 0.0]), degrees=True).as_matrix()
        gf_possible_target_pose_list = []
        distances_to_current = []
        gf_claw_pose = get_site_tmat(sim_node.mj_data, "endpoint")
        for grasp_pose_dict in grasping_poses:
            lf_grasp_position = np.asarray(grasp_pose_dict['p'])
            lf_grasp_q = np.asarray(grasp_pose_dict['r'])  # qw, qx, qy, qz
            lf_grasp_R = Rotation.from_quat(lf_grasp_q, scalar_first=True).as_matrix()
            lf_claw_mid_grasp_R = R_grasp_pose_claw @ lf_grasp_R # Object frame grasp orientation of claw mid point
            lf_claw_mid_grasp_pose = np.eye(4)
            lf_claw_mid_grasp_pose[:3, :3] = lf_claw_mid_grasp_R
            grasp_opening = grasp_pose_dict['opening']
            # Let's check if this opening is even feasible for the SO101 gripper through a very rough metric
            grasp_angle = np.arccos((2 * gripper_jaw_length**2 - grasp_opening**2) / (2 * gripper_jaw_length**2))
            if grasp_angle > 1.74533 or grasp_angle < -0.17453:  # 100 degrees in radians or less than -10 degrees
                continue
            lf_claw_mid_grasp_pose[:3, 3] = lf_grasp_position
            gf_claw_mid_grasp_pose = tmat_object @ lf_claw_mid_grasp_pose # This is thse pose midpoint of claw should be in world frame
            # Assuming the the claw's grasping plane XZ is always parallel during the grasp
            gf_claw_frame_grasp_pose = gf_claw_mid_grasp_pose.copy()
            gf_claw_frame_grasp_pose[:3, 3] -= gf_claw_frame_grasp_pose[:3, :3] @ np.array([0.0, 0.0, grasp_opening / 2.0 ])

            # Now we get the current pose of the claw and check if the IK can reach this pose
            distances_to_current.append(np.linalg.norm(gf_claw_frame_grasp_pose[:3, 3] - gf_claw_pose[:3, 3]))
            gf_possible_target_pose_list.append(gf_claw_frame_grasp_pose.copy())

        gf_feasible_target_pose_list = []
        print(f"Total possible grasping poses: {len(gf_possible_target_pose_list)}")
        if len(gf_possible_target_pose_list) > 0:
            # Start by getting the closest pose first
            sorted_indices = np.argsort(distances_to_current)
            for idx in sorted_indices:
                gf_target_pose = gf_possible_target_pose_list[idx]
                target_control = sim_node.target_control.copy()
                arm_joints = sim_node.nj - 1
                target_control[:arm_joints], converge = arm_ik.solve_ik(
                    gf_target_pose[:3, 3], 
                    gf_target_pose[:3, :3], 
                    sim_node.mj_data.qpos[:arm_joints]
                )
                print(f"Grasp check, pose: {gf_target_pose[:3,3]}, distance to current: {distances_to_current[idx]:.4f}, target_control: {target_control}, converge: {converge}")
                if converge:
                    gf_feasible_target_pose_list.append(target_control.copy())
                    break
        if not gf_feasible_target_pose_list:
            print("No feasible grasping pose found, just trying to close the gripper in current pose.")
            target_control = sim_node.target_control.copy()
            target_control[sim_node.nj - 1] = self.params.get('position', -0.17453)
        else:
            print(f"Feasible grasping pose found: {gf_feasible_target_pose_list[0]}, moving to that pose and closing the gripper.")
            target_control = gf_feasible_target_pose_list[0]
            target_control[sim_node.nj - 1] = self.params.get('position', -0.17453)
        return target_control


class ReleasePrimitive(MotionPrimitive):
    """
    Open gripper to release an object.
    
    YAML params:
        position: float - Gripper open position (default: 0.04)
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        target_control = sim_node.target_control.copy()
        target_control[sim_node.nj - 1] = self.params.get('position', 0.04)
        return target_control


class DelayPrimitive(MotionPrimitive):
    """
    Wait for a specified duration.
    
    YAML params:
        duration: float - Delay duration in seconds
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        duration = self.params['duration']
        sim_node.delay_cnt = int(duration / sim_node.delta_t)
        return sim_node.target_control.copy()


class OffsetCurrentPosePrimitive(MotionPrimitive):
    """
    Move relative to the current end-effector pose.
    
    YAML params:
        offset: [x, y, z] - Offset in meters
        gripper: float - Optional gripper position
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        offset = np.array(self.params['offset'])
        gripper = self.params.get('gripper', None)
        
        # Get current end-effector pose
        current_tmat = get_site_tmat(sim_node.mj_data, "endpoint") 
        
        # Apply offset in world frame
        target_tmat = current_tmat.copy()
        target_tmat[:3, 3] += offset
        
        tmat_target_local = tmat_armbase_2_world @ target_tmat
        
        target_control = sim_node.target_control.copy()
        arm_joints = sim_node.nj - 1
        target_control[:arm_joints] = arm_ik.properIK(
            tmat_target_local[:3, 3], 
            tmat_target_local[:3, :3], 
            sim_node.mj_data.qpos[:arm_joints]
        )
        
        if gripper is not None:
            target_control[sim_node.nj - 1] = gripper
        
        return target_control


class TrackObjectWithOffsetPrimitive(MotionPrimitive):
    """
    Move to an object with offset in the object's local frame.
    
    YAML params:
        object: str - Object body name
        local_offset: [x, y, z] - Offset in object's local frame
        rotation: dict - Optional rotation relative to object
        gripper: float - Gripper position
    """
    
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        object_name = self.params['object']
        local_offset = np.array(self.params.get('local_offset', [0, 0, 0]))
        rotation = self.params.get('rotation', None)
        gripper = self.params.get('gripper', None)
        
        tmat_object = get_body_tmat(sim_node.mj_data, object_name)
        
        # Apply offset in object's local frame
        offset_world = tmat_object[:3, :3] @ local_offset
        tmat_target = tmat_object.copy()
        tmat_target[:3, 3] += offset_world
        
        # Apply rotation if specified
        if rotation:
            if 'euler' in rotation:
                rot = Rotation.from_euler(rotation.get('seq', 'xyz'), 
                                         rotation['euler'], 
                                         degrees=rotation.get('degrees', False))
                tmat_target[:3, :3] = tmat_target[:3, :3] @ rot.as_matrix()
        
        tmat_target_local = tmat_armbase_2_world @ tmat_target
        
        target_control = sim_node.target_control.copy()
        arm_joints = sim_node.nj - 1
        target_control[:arm_joints] = arm_ik.properIK(
            tmat_target_local[:3, 3], 
            tmat_target_local[:3, :3], 
            sim_node.mj_data.qpos[:arm_joints]
        )
        
        if gripper is not None:
            target_control[sim_node.nj - 1] = gripper
        
        return target_control


# Registry of available motion primitives
MOTION_PRIMITIVES = {
    'move_to_object': MoveToObjectPrimitive,
    'move_to_position': MoveToPositionPrimitive,
    'grasp': GraspPrimitive,
    'release': ReleasePrimitive,
    'delay': DelayPrimitive,
    'offset_current': OffsetCurrentPosePrimitive,
    'track_object_offset': TrackObjectWithOffsetPrimitive,
}


def create_primitive(primitive_config):
    """
    Create a motion primitive from configuration.
    
    Args:
        primitive_config: dict with 'type' and 'params' keys
    
    Returns:
        MotionPrimitive instance
    """
    primitive_type = primitive_config['type']
    params = primitive_config.get('params', {})
    
    if primitive_type not in MOTION_PRIMITIVES:
        raise ValueError(f"Unknown motion primitive type: {primitive_type}")
    
    primitive_class = MOTION_PRIMITIVES[primitive_type]
    return primitive_class(primitive_type, params)
