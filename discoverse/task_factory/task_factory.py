"""
Task Factory for DISCOVERSE

This module provides a factory class that generates task classes from YAML configuration files.
The generated tasks include robot configuration, domain randomization, state machine execution,
success checking, and data recording.
"""

import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

import discoverse
from discoverse.envs import make_env
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.utils import get_body_tmat, SimpleStateMachine, step_func
from discoverse.task_factory.motion_primitives import create_primitive

# Robot configuration mapping
ROBOT_CONFIG_MAP = {
    'airbot_play': {
        'cfg_class': 'AirbotPlayCfg',
        'base_class': 'AirbotPlayBase',
        'task_base_class': 'AirbotPlayTaskBase',
        'ik_class': 'AirbotPlayIK',
        'cfg_module': 'discoverse.robots_env.airbot_play_base',
        'task_module': 'discoverse.task_base.airbot_task_base',
        'ik_module': 'discoverse.robots.airbot_play.airbot_play_ik',
        'recorder_func': 'recoder_airbot_play',
        'encoder_class': 'PyavImageEncoder',
    },
    'so101': {
        'cfg_class': 'SO101Cfg',
        'base_class': 'SO101Base',
        'task_base_class': 'SO101TaskBase',
        'ik_class': 'SO101_IK',
        'cfg_module': 'discoverse.robots_env.so101_base',
        'task_module': 'discoverse.task_base.so101_task_base',
        'ik_module': 'discoverse.robots.so101.so101_ik',
        'recorder_func': 'recoder_so101',
        'encoder_class': 'PyavImageEncoder',
    },
    'mmk2': {
        'cfg_class': 'MMK2Cfg',
        'base_class': 'MMK2Base',
        'task_base_class': 'MMK2TaskBase',
        'ik_class': 'MMK2IK',
        'cfg_module': 'discoverse.robots_env.mmk2_base',
        'task_module': 'discoverse.task_base.mmk2_task_base',
        'ik_module': 'discoverse.robots.mmk2.mmk2_ik',
        'recorder_func': 'recoder_airbot_play',
        'encoder_class': 'PyavImageEncoder',
    },
}
def _get_robot_classes(robot_name):
    """Dynamically import and return robot-specific classes."""
    if robot_name not in ROBOT_CONFIG_MAP:
        raise ValueError(f"Unsupported robot: {robot_name}. Supported: {list(ROBOT_CONFIG_MAP.keys())}")
    
    robot_info = ROBOT_CONFIG_MAP[robot_name]
    
    # Import configuration module
    cfg_module = __import__(robot_info['cfg_module'], fromlist=[robot_info['cfg_class']])
    cfg_class = getattr(cfg_module, robot_info['cfg_class'])
    
    # Import task base module
    task_module = __import__(robot_info['task_module'], fromlist=[robot_info['task_base_class']])
    task_base_class = getattr(task_module, robot_info['task_base_class'])
    
    # Import recorder function - handle different recorder modules
    if robot_name == 'so101':
        from discoverse.task_base.so101_task_base import recoder_so101 as recorder_func
    else:
        from discoverse.task_base.airbot_task_base import recoder_airbot_play as recorder_func
    
    # Import encoder class - all use PyavImageEncoder from airbot_task_base
    from discoverse.task_base.airbot_task_base import PyavImageEncoder
    
    return cfg_class, task_base_class, recorder_func, PyavImageEncoder
    return cfg_class, task_base_class, recoder_airbot_play, PyavImageEncoder

def _get_robot_ik(robot_name):
    """Dynamically import and return robot-specific IK class."""
    if robot_name not in ROBOT_CONFIG_MAP:
        raise ValueError(f"Unsupported robot: {robot_name}")
    
    robot_info = ROBOT_CONFIG_MAP[robot_name]
    ik_module = __import__(robot_info['ik_module'], fromlist=[robot_info['ik_class']])
    ik_class = getattr(ik_module, robot_info['ik_class'])
    return ik_class


class YAMLTaskFactory:
    """
    Factory class that creates task classes from YAML configuration files.
    
    The YAML file should contain:
        - task_name: Name of the task
        - robot_name: Robot type (e.g., 'airbot_play')
        - description: Task description
        - objects: List of objects in the scene
        - robot_config: Robot configuration parameters
        - randomization: Domain randomization parameters
        - action_sequence: List of motion primitives to execute
        - success_conditions: Conditions to determine task success
    """
    
    def __init__(self, yaml_path):
        """
        Initialize the factory with a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.task_name = self.config['task_name']
        self.robot_name = self.config.get('robot_name', 'airbot_play')
        
        # Parse motion primitives
        self.motion_primitives = []
        for action in self.config.get('action_sequence', []):
            primitive = create_primitive(action)
            self.motion_primitives.append(primitive)
    
    def create_robot_config(self):
        """
        Create and configure the robot configuration class based on YAML settings.
        
        Returns:
            Robot configuration object (e.g., AirbotPlayCfg, SO101Cfg)
        """
        # Dynamically get the correct config class for the robot
        cfg_class, _, _, _ = _get_robot_classes(self.robot_name)
        cfg = cfg_class()
        
        robot_config = self.config.get('robot_config', {})
        
        # Set Gaussian Splatting models
        if 'gs_models' in robot_config:
            for key, value in robot_config['gs_models'].items():
                cfg.gs_model_dict[key] = value
        
        # Set initial joint positions - length should match robot's nj
        if 'init_qpos' in robot_config:
            init_qpos = robot_config['init_qpos']
            if len(init_qpos) != len(cfg.init_qpos):
                raise ValueError(f"init_qpos length ({len(init_qpos)}) does not match robot's joint count ({len(cfg.init_qpos)})")
            cfg.init_qpos[:] = init_qpos
        
        # Set MJCF file path
        cfg.mjcf_file_path = robot_config.get(
            'mjcf_file_path', 
            f"mjcf/tmp/{self.robot_name}_{self.task_name}.xml"
        )
        
        # Set object list
        if 'objects' in self.config:
            cfg.obj_list = self.config['objects']
        
        # Set simulation parameters
        sim_params = robot_config.get('simulation', {})
        cfg.timestep = sim_params.get('timestep', 1/240)
        cfg.decimation = sim_params.get('decimation', 4)
        cfg.sync = sim_params.get('sync', True)
        cfg.headless = sim_params.get('headless', False)
        
        # Set rendering parameters
        render_params = robot_config.get('rendering', {})
        cfg.render_set = {
            'fps': render_params.get('fps', 20),
            'width': render_params.get('width', 640),
            'height': render_params.get('height', 480)
        }
        
        # Set camera IDs
        cfg.obs_rgb_cam_id = robot_config.get('camera_ids', [0, 1])
        
        # Set save options
        cfg.save_mjb_and_task_config = robot_config.get('save_mjb_and_task_config', True)
        
        return cfg
    
    def create_task_class(self):
        """
        Create a task class dynamically based on the YAML configuration.
        
        Returns:
            Task class (subclass of robot-specific TaskBase)
        """
        factory = self
        
        # Get the correct task base class for the robot
        _, TaskBaseClass, _, _ = _get_robot_classes(self.robot_name)
        
        class GeneratedTask(TaskBaseClass):
            """Dynamically generated task class from YAML configuration"""
            
            def __init__(self, cfg):
                super().__init__(cfg)
                self.factory_config = factory.config
                self.motion_primitives = factory.motion_primitives
            
            def domain_randomization(self):
                """Apply domain randomization based on YAML configuration"""
                randomization = self.factory_config.get('randomization', {})
                
                # Randomize object positions
                for obj_config in randomization.get('object_positions', []):
                    obj_name = obj_config['object']
                    position_range = obj_config.get('position_range', {})
                    
                    if 'x' in position_range:
                        x_range = position_range['x']
                        self.object_pose(obj_name)[0] += 2.0 * (np.random.random() - 0.5) * x_range
                    
                    if 'y' in position_range:
                        y_range = position_range['y']
                        self.object_pose(obj_name)[1] += 2.0 * (np.random.random() - 0.5) * y_range
                    
                    if 'z' in position_range:
                        z_range = position_range['z']
                        self.object_pose(obj_name)[2] += 2.0 * (np.random.random() - 0.5) * z_range
                
                # Randomize table height
                if randomization.get('table_height', False):
                    table_config = randomization.get('table_config', {})
                    obj_list = table_config.get('affected_objects', [])
                    table_name = table_config.get('table_name', 'table')
                    self.random_table_height(table_name=table_name, obj_name_list=obj_list)
                
                # Randomize table texture
                if randomization.get('table_texture', False):
                    self.random_table_texture()
                
                # Randomize object materials
                for material_name in randomization.get('materials', []):
                    self.random_material(material_name)
                
                # Randomize lighting
                if randomization.get('lighting', False):
                    light_config = randomization.get('light_config', {})
                    self.random_light(
                        random_dir=light_config.get('direction', True),
                        random_color=light_config.get('color', True),
                        random_active=light_config.get('active', True)
                    )
            
            def check_success(self):
                """Check task success based on YAML-defined conditions"""
                success_conditions = self.factory_config.get('success_conditions', {})
                
                all_conditions_met = True
                
                # Check position-based conditions
                for condition in success_conditions.get('position_conditions', []):
                    condition_type = condition['type']
                    
                    if condition_type == 'object_near_object':
                        obj1 = condition['object1']
                        obj2 = condition['object2']
                        threshold = condition['threshold']
                        
                        tmat1 = get_body_tmat(self.mj_data, obj1)
                        tmat2 = get_body_tmat(self.mj_data, obj2)
                        
                        distance = np.linalg.norm(tmat1[:3, 3] - tmat2[:3, 3])
                        
                        if distance > threshold:
                            all_conditions_met = False
                            break
                    
                    elif condition_type == 'object_at_position':
                        obj = condition['object']
                        target_pos = np.array(condition['position'])
                        threshold = condition['threshold']
                        
                        tmat = get_body_tmat(self.mj_data, obj)
                        distance = np.linalg.norm(tmat[:3, 3] - target_pos)
                        
                        if distance > threshold:
                            all_conditions_met = False
                            break
                    
                    elif condition_type == 'planar_distance':
                        obj1 = condition['object1']
                        obj2 = condition['object2']
                        threshold = condition['threshold']
                        axes = condition.get('axes', 'xy')
                        
                        tmat1 = get_body_tmat(self.mj_data, obj1)
                        tmat2 = get_body_tmat(self.mj_data, obj2)
                        
                        if axes == 'xy':
                            distance = np.hypot(tmat1[0, 3] - tmat2[0, 3], 
                                              tmat1[1, 3] - tmat2[1, 3])
                        elif axes == 'xz':
                            distance = np.hypot(tmat1[0, 3] - tmat2[0, 3], 
                                              tmat1[2, 3] - tmat2[2, 3])
                        elif axes == 'yz':
                            distance = np.hypot(tmat1[1, 3] - tmat2[1, 3], 
                                              tmat1[2, 3] - tmat2[2, 3])
                        
                        if distance > threshold:
                            all_conditions_met = False
                            break
                
                # Check orientation-based conditions
                for condition in success_conditions.get('orientation_conditions', []):
                    condition_type = condition['type']
                    
                    if condition_type == 'object_upright':
                        obj = condition['object']
                        axis = condition.get('axis', 'z')
                        threshold = condition.get('threshold', 0.99)
                        
                        tmat = get_body_tmat(self.mj_data, obj)
                        
                        if axis == 'z':
                            if abs(tmat[2, 2]) < threshold:
                                all_conditions_met = False
                                break
                        elif axis == 'y':
                            if abs(tmat[1, 1]) < threshold:
                                all_conditions_met = False
                                break
                        elif axis == 'x':
                            if abs(tmat[0, 0]) < threshold:
                                all_conditions_met = False
                                break
                
                # Check custom conditions (using eval - be careful!)
                for condition in success_conditions.get('custom_conditions', []):
                    expression = condition['expression']
                    # Provide context for evaluation
                    context = {
                        'np': np,
                        'get_body_tmat': get_body_tmat,
                        'mj_data': self.mj_data,
                        'abs': abs,
                    }
                    try:
                        if not eval(expression, context):
                            all_conditions_met = False
                            break
                    except Exception as e:
                        print(f"Error evaluating condition: {expression}, {e}")
                        all_conditions_met = False
                        break
                
                return all_conditions_met
        
        return GeneratedTask
    
    def setup_environment(self):
        """
        Setup the environment by exporting the MJCF file.
        
        Returns:
            Configuration object
        """
        cfg = self.create_robot_config()
        
        # Create environment and export XML
        env = make_env(self.robot_name, self.task_name)
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path)
        env.export_xml(mjcf_path)
        
        return cfg
    
    def create_state_machine_executor(self, sim_node, arm_ik):
        """
        Create a state machine executor for the motion primitives.
        
        Args:
            sim_node: The simulation node
            arm_ik: The inverse kinematics solver
        
        Returns:
            Function that executes the state machine
        """
        tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))
        
        def execute_state(stm):
            """Execute the current state in the state machine"""
            if stm.state_idx >= len(self.motion_primitives):
                return sim_node.target_control.copy()
            
            print(f"Executing state {stm.state_idx}: {self.motion_primitives[stm.state_idx].name}")
            primitive = self.motion_primitives[stm.state_idx]
            target_control = primitive.execute(sim_node, arm_ik, tmat_armbase_2_world)
            sim_node.target_control[:] = target_control
            
            # Set joint move ratio - use robot's nj dynamically
            dif = np.abs(sim_node.target_control - sim_node.mj_data.ctrl[:sim_node.nj])
            sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            
            return target_control
        
        return execute_state


def load_task_from_yaml(yaml_path):
    """
    Convenience function to load a task from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
    
    Returns:
        Tuple of (TaskClass, config, factory)
    """
    factory = YAMLTaskFactory(yaml_path)
    cfg = factory.setup_environment()
    TaskClass = factory.create_task_class()
    
    return TaskClass, cfg, factory


def run_yaml_task(yaml_path, data_idx=0, data_set_size=1, auto=False, headless=False, use_gs=False):
    """
    Run a task defined by a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        data_idx: Starting data index
        data_set_size: Number of data samples to collect
        auto: Whether to run in automatic mode (headless + no sync)
        headless: Whether to run without visualization
        use_gs: Whether to use Gaussian Splatting renderer
    """
    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    # Load task from YAML
    TaskClass, cfg, factory = load_task_from_yaml(yaml_path)
    
    # Get robot-specific classes (including recorder and encoder)
    _, _, recorder_func, EncoderClass = _get_robot_classes(factory.robot_name)
    
    # Update config based on arguments
    if auto:
        cfg.headless = True
        cfg.sync = False
    elif headless:
        cfg.headless = True
    else:
        # Ensure visualization is enabled by default
        cfg.headless = False
        cfg.sync = True
    cfg.use_gaussian_renderer = use_gs
    
    # Setup save directory
    task_name = factory.task_name
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", task_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create simulation node
    sim_node = TaskClass(cfg)
    
    # Save model and config
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        import mujoco
        mjb_path = os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb"))
        mujoco.mj_saveModel(sim_node.mj_model, mjb_path)
        
        # Save YAML config
        import shutil
        shutil.copy(yaml_path, os.path.join(save_dir, os.path.basename(yaml_path)))
    
    # Import robot IK dynamically
    IKClass = _get_robot_ik(factory.robot_name)
    arm_ik = IKClass()
    
    # Create state machine executor
    execute_state = factory.create_state_machine_executor(sim_node, arm_ik)
    
    # Setup state machine
    stm = SimpleStateMachine()
    stm.max_state_cnt = len(factory.motion_primitives) - 1
    max_time = factory.config.get('max_time', 20.0)
    primitive_timeout = factory.config.get('primitive_timeout', 4.0)  # Default 4 second timeout per primitive
    
    # Initialize action array with robot's joint count (nj)
    action = np.zeros(sim_node.nj)
    move_speed = factory.config.get('move_speed', 0.75)
    
    data_end_idx = data_idx + data_set_size
    
    sim_node.reset()
    primitive_start_time = 0.0
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {
                cam_id: EncoderClass(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) 
                for cam_id in cfg.obs_rgb_cam_id
            }
            primitive_start_time = sim_node.mj_data.time
        
        try:
            if stm.trigger():
                execute_state(stm)
                primitive_start_time = sim_node.mj_data.time  # Reset timer when new primitive starts
            else:
                stm.update()
            
            # Check for primitive timeout
            if sim_node.mj_data.time - primitive_start_time > primitive_timeout:
                print(f"Primitive {stm.state_idx} timed out after {primitive_timeout}s, moving to next state")
                stm.next()
                primitive_start_time = sim_node.mj_data.time
            
            if sim_node.checkActionDone():
                stm.next()
                primitive_start_time = sim_node.mj_data.time  # Reset timer when moving to next primitive
        
        except ValueError as ve:
            print(f"IK error: {ve}")
            sim_node.reset()
        
        # Smooth action interpolation - use robot's nj dynamically
        for i in range(sim_node.nj - 1):
            action[i] = step_func(
                action[i], 
                sim_node.target_control[i], 
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t
            )
        action[sim_node.nj - 1] = sim_node.target_control[sim_node.nj - 1]
        
        obs, _, _, _, _ = sim_node.step(action)
        
        # Record data
        try:
            if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                imgs = obs.pop("img")
                for cam_id, img in imgs.items():
                    encoders[cam_id].encode(img, obs["time"])
                act_lst.append(action.tolist().copy())
                obs_lst.append(obs)
        except Exception as ex:
                    print(f"cam_id: {cam_id}{str(ex)}")
        
        # Check completion
        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recorder_func(save_path, act_lst, obs_lst, cfg)
                for encoder in encoders.values():
                    encoder.close()
                data_idx += 1
                print(f"\r{data_idx:4}/{data_set_size:4} ", end="")
                if data_idx >= data_end_idx:
                    break
            else:
                print(f"{data_idx} Failed")
            
            sim_node.reset()
