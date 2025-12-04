"""
Utility functions for YAML task configuration validation and debugging.
"""

import yaml
import numpy as np
from typing import Dict, List, Any


class YAMLConfigValidator:
    """Validator for YAML task configuration files"""
    
    REQUIRED_FIELDS = ['task_name', 'robot_name', 'robot_config', 'action_sequence', 'success_conditions']
    REQUIRED_ROBOT_CONFIG_FIELDS = ['init_qpos', 'simulation', 'rendering']
    VALID_PRIMITIVE_TYPES = [
        'move_to_object', 'move_to_position', 'grasp', 'release', 
        'delay', 'offset_current', 'track_object_offset'
    ]
    # Robot joint counts for validation
    ROBOT_JOINT_COUNTS = {
        'airbot_play': 7,
        'so101': 6,
        'mmk2': 12,
        'tok2': 7,
        'leaphand': 16,
        'hand_with_arm': 18,
    }
    
    def __init__(self, config_path: str):
        """
        Initialize validator with a YAML config file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """
        Validate the YAML configuration.
        
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Check required top-level fields
        for field in self.REQUIRED_FIELDS:
            if field not in self.config:
                self.errors.append(f"Missing required field: {field}")
        
        # Validate robot config
        if 'robot_config' in self.config:
            self._validate_robot_config()
        
        # Validate action sequence
        if 'action_sequence' in self.config:
            self._validate_action_sequence()
        
        # Validate success conditions
        if 'success_conditions' in self.config:
            self._validate_success_conditions()
        
        # Validate randomization if present
        if 'randomization' in self.config:
            self._validate_randomization()
        
        return len(self.errors) == 0
    
    def _validate_robot_config(self):
        """Validate robot configuration section"""
        robot_config = self.config['robot_config']
        
        for field in self.REQUIRED_ROBOT_CONFIG_FIELDS:
            if field not in robot_config:
                self.errors.append(f"Missing required robot_config field: {field}")
        
        # Check init_qpos - validate against robot's joint count
        if 'init_qpos' in robot_config:
            qpos = robot_config['init_qpos']
            if not isinstance(qpos, list):
                self.errors.append("init_qpos must be a list")
            else:
                robot_name = self.config.get('robot_name', 'airbot_play')
                expected_nj = self.ROBOT_JOINT_COUNTS.get(robot_name)
                if expected_nj is not None:
                    if len(qpos) != expected_nj:
                        self.errors.append(f"init_qpos length ({len(qpos)}) does not match {robot_name}'s joint count ({expected_nj})")
                else:
                    self.warnings.append(f"Unknown robot type '{robot_name}', cannot validate init_qpos length")
        
        # Check simulation params
        if 'simulation' in robot_config:
            sim = robot_config['simulation']
            if 'timestep' not in sim:
                self.warnings.append("simulation.timestep not specified, using default")
    
    def _validate_action_sequence(self):
        """Validate action sequence primitives"""
        actions = self.config['action_sequence']
        
        if not isinstance(actions, list):
            self.errors.append("action_sequence must be a list")
            return
        
        if len(actions) == 0:
            self.warnings.append("action_sequence is empty")
        
        for i, action in enumerate(actions):
            if 'type' not in action:
                self.errors.append(f"Action {i} missing 'type' field")
                continue
            
            action_type = action['type']
            if action_type not in self.VALID_PRIMITIVE_TYPES:
                self.errors.append(f"Action {i} has invalid type: {action_type}")
            
            # Validate specific primitive parameters
            params = action.get('params', {})
            self._validate_primitive_params(action_type, params, i)
    
    def _validate_primitive_params(self, action_type: str, params: Dict, index: int):
        """Validate parameters for specific primitive types"""
        if action_type == 'move_to_object':
            if 'object' not in params:
                self.errors.append(f"Action {index} (move_to_object) missing 'object' parameter")
        
        elif action_type == 'move_to_position':
            if 'position' not in params:
                self.errors.append(f"Action {index} (move_to_position) missing 'position' parameter")
            elif not isinstance(params['position'], list) or len(params['position']) != 3:
                self.errors.append(f"Action {index} position must be [x, y, z]")
        
        elif action_type == 'delay':
            if 'duration' not in params:
                self.errors.append(f"Action {index} (delay) missing 'duration' parameter")
        
        elif action_type == 'offset_current':
            if 'offset' not in params:
                self.errors.append(f"Action {index} (offset_current) missing 'offset' parameter")
            elif not isinstance(params['offset'], list) or len(params['offset']) != 3:
                self.errors.append(f"Action {index} offset must be [x, y, z]")
        
        elif action_type == 'track_object_offset':
            if 'object' not in params:
                self.errors.append(f"Action {index} (track_object_offset) missing 'object' parameter")
    
    def _validate_success_conditions(self):
        """Validate success conditions"""
        success = self.config['success_conditions']
        
        if not isinstance(success, dict):
            self.errors.append("success_conditions must be a dictionary")
            return
        
        # Check if at least one condition type is present
        condition_types = ['position_conditions', 'orientation_conditions', 'custom_conditions']
        has_conditions = any(ct in success for ct in condition_types)
        
        if not has_conditions:
            self.warnings.append("No success conditions defined")
    
    def _validate_randomization(self):
        """Validate randomization configuration"""
        randomization = self.config['randomization']
        
        if 'object_positions' in randomization:
            for obj_config in randomization['object_positions']:
                if 'object' not in obj_config:
                    self.warnings.append("Object position randomization missing 'object' field")
    
    def print_report(self):
        """Print validation report"""
        print(f"\n{'='*60}")
        print(f"YAML Configuration Validation Report")
        print(f"File: {self.config_path}")
        print(f"{'='*60}\n")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("✓ Configuration is valid!\n")
            return
        
        if self.errors:
            print(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")
            print()
        
        if self.warnings:
            print(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
            print()
        
        if len(self.errors) == 0:
            print("Configuration is valid (with warnings)\n")
        else:
            print("Configuration is INVALID\n")


def validate_yaml_config(config_path: str) -> bool:
    """
    Convenience function to validate a YAML config and print report.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        True if valid, False otherwise
    """
    validator = YAMLConfigValidator(config_path)
    is_valid = validator.validate()
    validator.print_report()
    return is_valid


def summarize_task(config_path: str):
    """
    Print a human-readable summary of a task configuration.
    
    Args:
        config_path: Path to YAML configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Task Summary: {config.get('task_name', 'Unknown')}")
    print(f"{'='*60}\n")
    
    print(f"Description: {config.get('description', 'No description')}")
    print(f"Robot: {config.get('robot_name', 'Unknown')}\n")
    
    # Objects
    objects = config.get('objects', [])
    print(f"Objects ({len(objects)}):")
    for obj in objects:
        print(f"  - {obj}")
    print()
    
    # Action sequence
    actions = config.get('action_sequence', [])
    print(f"Action Sequence ({len(actions)} steps):")
    for i, action in enumerate(actions):
        action_type = action.get('type', 'unknown')
        params = action.get('params', {})
        
        # Format params briefly
        param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:2]])
        if len(params) > 2:
            param_str += ", ..."
        
        print(f"  {i:2}. {action_type:20} ({param_str})")
    print()
    
    # Success conditions
    success = config.get('success_conditions', {})
    total_conditions = sum([
        len(success.get('position_conditions', [])),
        len(success.get('orientation_conditions', [])),
        len(success.get('custom_conditions', []))
    ])
    print(f"Success Conditions: {total_conditions} checks")
    
    # Randomization
    rand = config.get('randomization', {})
    rand_enabled = []
    if rand.get('table_height'): rand_enabled.append('table_height')
    if rand.get('table_texture'): rand_enabled.append('table_texture')
    if rand.get('lighting'): rand_enabled.append('lighting')
    if rand.get('object_positions'): rand_enabled.append(f"{len(rand['object_positions'])} object positions")
    if rand.get('materials'): rand_enabled.append(f"{len(rand['materials'])} materials")
    
    print(f"Randomization: {', '.join(rand_enabled) if rand_enabled else 'None'}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yaml_utils.py <config.yaml> [--validate|--summarize]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--validate"
    
    if mode == "--validate":
        validate_yaml_config(config_path)
    elif mode == "--summarize":
        summarize_task(config_path)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
