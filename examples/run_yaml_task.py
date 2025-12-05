"""
Example script demonstrating how to use the YAML-based task factory.

Usage:
    # Run with visualization (default)
    python run_yaml_task.py --config configs/tasks/cover_cup_yaml.yaml
    
    # Run without visualization
    python run_yaml_task.py --config configs/tasks/cover_cup_yaml.yaml --headless
    
    # Collect multiple samples in automatic mode (headless + fast)
    python run_yaml_task.py --config configs/tasks/cover_cup_yaml.yaml --data_idx 0 --data_set_size 10 --auto
    
    # Run with Gaussian Splatting renderer
    python run_yaml_task.py --config configs/tasks/pick_place_block.yaml --use_gs
"""

import os
import argparse
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.task_factory.task_factory import run_yaml_task


def main():
    parser = argparse.ArgumentParser(description="Run a task defined by a YAML configuration file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--data_idx", type=int, default=0, help="Starting data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="Number of data samples to collect")
    parser.add_argument("--auto", action="store_true", help="Run in automatic mode (headless, no sync)")
    parser.add_argument("--headless", action="store_true", help="Run without visualization (headless mode)")
    parser.add_argument("--use_gs", action="store_true", help="Use Gaussian Splatting renderer")
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to DISCOVERSE root
        config_path = os.path.join(DISCOVERSE_ROOT_DIR, config_path)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    print(f"Loading task from: {config_path}")
    
    # Run the task
    run_yaml_task(
        yaml_path=config_path,
        data_idx=args.data_idx,
        data_set_size=args.data_set_size,
        auto=args.auto,
        headless=args.headless,
        use_gs=args.use_gs
    )


if __name__ == "__main__":
    main()
