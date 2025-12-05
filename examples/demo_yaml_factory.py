#!/usr/bin/env python3
"""
Demo script showing various ways to use the YAML Task Factory.

This script demonstrates:
1. Loading and validating a YAML task
2. Summarizing task configuration
3. Running a task programmatically
4. Creating a task class dynamically
"""

import os
import sys
import numpy as np

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.task_factory import (
    YAMLTaskFactory, 
    load_task_from_yaml,
    validate_yaml_config,
    summarize_task
)


def demo_validation():
    """Demo: Validate a YAML configuration"""
    print("\n" + "="*70)
    print("DEMO 1: Validating YAML Configuration")
    print("="*70)
    
    config_path = os.path.join(
        DISCOVERSE_ROOT_DIR, 
        "discoverse/configs/tasks/cover_cup_yaml.yaml"
    )
    
    print(f"\nValidating: {config_path}")
    is_valid = validate_yaml_config(config_path)
    
    if is_valid:
        print("✓ Configuration is valid and ready to use!")
    else:
        print("✗ Configuration has errors. Please fix them before running.")
    
    return is_valid


def demo_summarize():
    """Demo: Summarize a task configuration"""
    print("\n" + "="*70)
    print("DEMO 2: Task Summary")
    print("="*70)
    
    config_path = os.path.join(
        DISCOVERSE_ROOT_DIR, 
        "discoverse/configs/tasks/pick_place_block.yaml"
    )
    
    summarize_task(config_path)


def demo_factory_basics():
    """Demo: Basic factory usage"""
    print("\n" + "="*70)
    print("DEMO 3: Factory Basics")
    print("="*70)
    
    config_path = os.path.join(
        DISCOVERSE_ROOT_DIR, 
        "discoverse/configs/tasks/cover_cup_yaml.yaml"
    )
    
    print(f"\nLoading task from: {config_path}")
    factory = YAMLTaskFactory(config_path)
    
    print(f"\nTask Name: {factory.task_name}")
    print(f"Robot Name: {factory.robot_name}")
    print(f"Number of Motion Primitives: {len(factory.motion_primitives)}")
    
    # Show motion primitive sequence
    print("\nMotion Primitive Sequence:")
    for i, primitive in enumerate(factory.motion_primitives):
        print(f"  {i:2}. {primitive.name:20} - {list(primitive.params.keys())[:3]}")
    
    # Create robot config
    print("\nCreating robot configuration...")
    cfg = factory.create_robot_config()
    print(f"  Initial joint positions: {cfg.init_qpos}")
    print(f"  Timestep: {cfg.timestep}")
    print(f"  Render resolution: {cfg.render_set['width']}x{cfg.render_set['height']}")
    print(f"  Camera IDs: {cfg.obs_rgb_cam_id}")
    
    # Create task class
    print("\nCreating task class...")
    TaskClass = factory.create_task_class()
    print(f"  Task Class: {TaskClass.__name__}")
    print(f"  Base Class: {TaskClass.__bases__[0].__name__}")


def demo_load_and_inspect():
    """Demo: Load task and inspect generated class"""
    print("\n" + "="*70)
    print("DEMO 4: Load and Inspect Generated Task Class")
    print("="*70)
    
    config_path = os.path.join(
        DISCOVERSE_ROOT_DIR, 
        "discoverse/configs/tasks/cover_cup_yaml.yaml"
    )
    
    print(f"\nLoading task from: {config_path}")
    TaskClass, cfg, factory = load_task_from_yaml(config_path)
    
    print(f"\nGenerated Task Class Info:")
    print(f"  Class name: {TaskClass.__name__}")
    print(f"  Module: {TaskClass.__module__}")
    print(f"  Docstring: {TaskClass.__doc__.strip() if TaskClass.__doc__ else 'None'}")
    
    # Show methods
    print(f"\nKey Methods:")
    methods = ['domain_randomization', 'check_success', '__init__']
    for method_name in methods:
        if hasattr(TaskClass, method_name):
            print(f"  ✓ {method_name}")
    
    print(f"\nConfiguration:")
    print(f"  Objects: {cfg.obj_list}")
    print(f"  Sync mode: {cfg.sync}")
    print(f"  Headless: {cfg.headless}")


def demo_motion_primitive_info():
    """Demo: Show information about motion primitives"""
    print("\n" + "="*70)
    print("DEMO 5: Motion Primitive Information")
    print("="*70)
    
    from discoverse.task_factory.motion_primitives import MOTION_PRIMITIVES
    
    print(f"\nAvailable Motion Primitives ({len(MOTION_PRIMITIVES)}):\n")
    
    for name, primitive_class in MOTION_PRIMITIVES.items():
        doc = primitive_class.__doc__
        # Extract first line of docstring
        if doc:
            first_line = doc.strip().split('\n')[0]
        else:
            first_line = "No description"
        
        print(f"  • {name:25} - {first_line}")
    
    print("\nMotion primitives use get_body_tmat() to track object poses")
    print("with optional offsets and rotations for precise manipulation.")


def demo_success_conditions():
    """Demo: Show success condition types"""
    print("\n" + "="*70)
    print("DEMO 6: Success Condition Types")
    print("="*70)
    
    print("\nSupported Success Condition Types:\n")
    
    conditions = {
        "Position Conditions": [
            "object_near_object - Check if two objects are close",
            "object_at_position - Check if object is at target position",
            "planar_distance - Check distance in specific plane (xy, xz, yz)",
        ],
        "Orientation Conditions": [
            "object_upright - Check if object maintains orientation",
        ],
        "Custom Conditions": [
            "expression - Evaluate Python expression (advanced)",
        ]
    }
    
    for category, cond_list in conditions.items():
        print(f"  {category}:")
        for cond in cond_list:
            print(f"    • {cond}")
        print()


def main():
    """Run all demos"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "YAML Task Factory - Demo Suite" + " "*23 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    demos = [
        ("Validation", demo_validation),
        ("Summary", demo_summarize),
        ("Factory Basics", demo_factory_basics),
        ("Load and Inspect", demo_load_and_inspect),
        ("Motion Primitives", demo_motion_primitive_info),
        ("Success Conditions", demo_success_conditions),
    ]
    
    # Check if specific demo requested
    if len(sys.argv) > 1:
        demo_num = int(sys.argv[1])
        if 1 <= demo_num <= len(demos):
            demos[demo_num - 1][1]()
            return
    
    # Run all demos
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Error in demo {i} ({name}): {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "All Demos Complete!" + " "*27 + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    print("To run a specific demo:")
    print("  python demo_yaml_factory.py <demo_number>")
    print("\nTo run a task:")
    print("  python run_yaml_task.py --config <path/to/config.yaml>")
    print("\nFor more information:")
    print("  See discoverse/task_factory/README.md")
    print()


if __name__ == "__main__":
    main()
