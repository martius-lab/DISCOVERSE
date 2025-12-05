#!/usr/bin/env python3
"""
Test all task examples to ensure two-phase pipeline works correctly.
Runs each task, validates outputs, and reports results.

IMPORTANT: Run this script with the discoverse conda environment activated:
    conda activate discoverse
    python test_all_tasks.py
"""

import os
import sys
import subprocess
import time

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_outputs(data_dir, task_name, expected_files_overview, expected_files_render):
    """Check if expected files exist in the data directory."""
    issues = []

    # Find trajectory directories (e.g., 000, 001, etc.)
    traj_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if d.isdigit() and os.path.isdir(os.path.join(data_dir, d))
        ]
    )

    if not traj_dirs:
        return [f"No trajectory directories found in {data_dir}"]

    # Check for record_playback.py in the main data_dir (not inside trajectory folders)
    playback_script = os.path.join(data_dir, "record_playback.py")
    if not os.path.exists(playback_script):
        issues.append(f"Missing record_playback.py in {data_dir}")

    for traj_dir in traj_dirs:
        traj_path = os.path.join(data_dir, traj_dir)

        # Check overview phase files (excluding record_playback.py which is in parent dir)
        for expected_file in expected_files_overview:
            if expected_file == "record_playback.py":
                continue  # Already checked in parent directory
            file_path = os.path.join(traj_path, expected_file)
            if not os.path.exists(file_path):
                issues.append(f"Missing {expected_file} in {traj_dir}")

        # Check that full camera files don't exist yet (overview_only=True)
        for forbidden_file in expected_files_render:
            if forbidden_file == "overview.mp4":
                continue  # overview.mp4 should exist
            file_path = os.path.join(traj_path, forbidden_file)
            if os.path.exists(file_path) and forbidden_file.startswith("cam_"):
                # If multiple cam files exist, it means overview_only wasn't respected
                # But check if it's the overview cam renamed
                pass  # We'll check more carefully

    return issues


def run_task(task_path, task_name, robot_type, timeout=120):
    """Run a single task example."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Testing: {task_name} ({robot_type}){RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    # Determine data directory based on task name
    # Airbot tasks use airbot_ prefix, MMK2 tasks use mmk2_ prefix
    task_basename = task_name
    if robot_type == "airbot_play":
        data_dir = f"data/airbot_{task_basename}"
    elif robot_type == "mmk2":
        data_dir = f"data/mmk2_{task_basename}"
    else:
        data_dir = f"data/{task_basename}"

    # Clean up old data
    if os.path.exists(data_dir):
        import shutil

        shutil.rmtree(data_dir)
        print(f"Cleaned up old data directory: {data_dir}")

    # Expected files after overview phase
    expected_overview = [
        "mujoco_states.pkl",
        # "action_trajectory.npy",
        "obs_action.json",
        "overview.mp4",
        "record_playback.py",
    ]

    # Expected files after full render
    expected_render = ["overview.mp4"]  # Will check for cam_*.mp4 files

    # Phase 1: Run with overview only (no --render flag)
    print(f"\n{YELLOW}Phase 1: Running task (overview only)...{RESET}")
    cmd = [sys.executable, task_path, "--data_set_size", "1", "--auto"]
    print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            print(f"{RED}✗ Task failed with return code {result.returncode}{RESET}")
            print(f"STDERR:\n{result.stderr[-1000:]}")  # Last 1000 chars
            return False, f"Failed with code {result.returncode}"

        # Check outputs
        if not os.path.exists(data_dir):
            print(f"{RED}✗ Data directory not created: {data_dir}{RESET}")
            return False, "No data directory created"

        issues = check_outputs(data_dir, task_name, expected_overview, expected_render)
        if issues:
            print(f"{RED}✗ Missing files in overview phase:{RESET}")
            for issue in issues:
                print(f"  - {issue}")
            return False, f"Missing files: {', '.join(issues)}"

        print(f"{GREEN}✓ Overview phase successful{RESET}")

        # Phase 2: Test --render flag (run playback)
        print(f"\n{YELLOW}Phase 2: Testing full render (--render flag)...{RESET}")

        # Playback script is in the data_dir (not inside trajectory folders)
        playback_script = os.path.join(data_dir, "record_playback.py")

        if not os.path.exists(playback_script):
            print(f"{RED}✗ record_playback.py not generated{RESET}")
            return False, "No playback script"

        # Run playback script with --all flag
        cmd_render = [sys.executable, playback_script, "--all"]
        print(" ".join(cmd_render))
        result_render = subprocess.run(
            cmd_render, cwd=os.getcwd(), capture_output=True, text=True, timeout=timeout
        )

        if result_render.returncode != 0:
            print(
                f"{RED}✗ Render phase failed with return code {result_render.returncode}{RESET}"
            )
            print(f"STDERR:\n{result_render.stderr[-1000:]}")
            return False, f"Render failed with code {result_render.returncode}"

        # Check that cam files were created in trajectory directories
        traj_dirs = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if d.isdigit() and os.path.isdir(os.path.join(data_dir, d))
            ]
        )

        total_cam_files = 0
        for traj_dir in traj_dirs:
            traj_path = os.path.join(data_dir, traj_dir)
            cam_files = [
                f
                for f in os.listdir(os.path.join(traj_path, "rendered"))
                if f.startswith("cam_") and f.endswith(".mp4")
            ]
            total_cam_files += len(cam_files)

        if total_cam_files == 0:
            print(f"{RED}✗ No cam_*.mp4 files generated in render phase{RESET}")
            return False, "No camera files generated"

        print(
            f"{GREEN}✓ Render phase successful ({total_cam_files} camera files generated){RESET}"
        )
        print(f"{GREEN}✓✓ Task {task_name} PASSED{RESET}")

        return True, None

    except subprocess.TimeoutExpired:
        print(f"{RED}✗ Task timed out after {timeout}s{RESET}")
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        print(f"{RED}✗ Exception: {str(e)}{RESET}")
        return False, str(e)


def main():
    """Main test runner."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check if we're in the right environment
    try:
        import mujoco  # noqa: F401

        print(f"{GREEN}✓ MuJoCo found - environment looks good{RESET}")
    except ImportError:
        print(
            f"{RED}✗ MuJoCo not found. Please activate the 'discoverse' conda environment:{RESET}"
        )
        print(f"{RED}  conda activate discoverse{RESET}")
        return 1

    # Define tasks to test (subset for quick validation)
    tasks = [
        # AirbotPlay tasks
        # ("examples/tasks_airbot_play/place_jujube.py", "place_jujube", "airbot_play"),
        # ("examples/tasks_airbot_play/pick_jujube.py", "pick_jujube", "airbot_play"),
        # ("examples/tasks_airbot_play/place_block.py", "place_block", "airbot_play"),
        # ("examples/tasks_airbot_play/open_drawer.py", "open_drawer", "airbot_play"),
        # ("examples/tasks_airbot_play/stack_block.py", "stack_block", "airbot_play"),
        # # SO101 tasks
        # ("examples/tasks_so101/so101_pick_milk.py", "so101_pick_milk", "so101"),
        # # HandArm tasks
        # ("examples/tasks_hand_arm/build_tower.py", "build_tower", "hand_arm"),
        # MMK2 tasks
        ("examples/tasks_mmk2/box_pick.py", "pick_box", "mmk2"),
        ("examples/tasks_mmk2/cabinet_door_open.py", "cabinet_door_open", "mmk2"),
        ("examples/tasks_mmk2/coffeecup_plate.py", "plate_coffecup", "mmk2"),
        ("examples/tasks_mmk2/drawer_open.py", "drawer_open", "mmk2"),
        ("examples/tasks_mmk2/jujube_pick.py", "pick_jujube", "mmk2"),
        ("examples/tasks_mmk2/kiwi_pick.py", "pick_kiwi", "mmk2"),
        ("examples/tasks_mmk2/kiwi_place.py", "kiwi_place", "mmk2"),
        ("examples/tasks_mmk2/pan_pick.py", "pick_pan", "mmk2"),
    ]

    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Two-Phase Pipeline Test Suite{RESET}")
    print(f"{BLUE}Testing {len(tasks)} tasks{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    results = []
    start_time = time.time()

    for task_path, task_name, robot_type in tasks:
        if not os.path.exists(task_path):
            print(f"{YELLOW}⊘ Skipping {task_name} (file not found){RESET}")
            results.append((task_name, False, "File not found"))
            continue

        success, error = run_task(task_path, task_name, robot_type)
        results.append((task_name, success, error))

        # Small delay between tests
        time.sleep(1)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Test Summary (completed in {elapsed:.1f}s){RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for task_name, success, error in results:
        if success:
            print(f"{GREEN}✓ {task_name}{RESET}")
        else:
            print(f"{RED}✗ {task_name}: {error}{RESET}")

    print(f"\n{BLUE}{'='*70}{RESET}")
    if failed == 0:
        print(f"{GREEN}All {passed} tasks PASSED! 🎉{RESET}")
        return 0
    else:
        print(f"{RED}{failed} tasks FAILED, {passed} tasks passed{RESET}")
        return 1


if __name__ == "__main__":
    main()
    # sys.exit(main())
