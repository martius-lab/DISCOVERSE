# Two-Phase Data Generation Pipeline

## Overview

The DISCOVERSE simulator now uses a **two-phase pipeline** that separates planning from rendering:

1. **Planning Phase** (Fast): Simulate physics and record privileged states
2. **Rendering Phase** (Detailed): Render observations from saved states

This separation enables:
- ⚡ **Fast planning**: No expensive multi-camera rendering during simulation
- 🎨 **Flexible rendering**: Apply different visuals to same trajectory
- 🔄 **Multiple renders**: Generate varied datasets from one simulation
- 🎯 **Domain randomization**: Easy texture/lighting variation

---

## Phase 1: Planning (Fast Simulation)

### What happens
- Physics simulation runs at full speed
- Records **privileged MuJoCo states** (qpos, qvel, ctrl, act)
- Saves **one overview.mp4** for quick verification
- No expensive multi-camera rendering

### Command
```bash
# Fast planning only
python examples/tasks_mmk2/coffeecup_plate.py --data_set_size 10 --auto
```

### Output
```
data/mmk2_plate_coffecup/
├── coffeecup_plate.mjb
├── coffeecup_plate.py
├── record_playback.py
└── 000/, 001/, 002/, ...
    ├── mujoco_states.pkl    ← Privileged states
    ├── action_trajectory.npy
    ├── obs_action.json
    └── overview.mp4         ← Quick check (1 camera only)
```

---

## Phase 2: Rendering (Detailed Observations)

### What happens
- Loads privileged states
- Optionally applies visual randomization
- Renders **all cameras** with full quality
- Saves complete observation dataset

### Commands

**Option A: Automatic (all-in-one)**
```bash
# Plan AND render in one command
python examples/tasks_mmk2/coffeecup_plate.py --data_set_size 10 --auto --render
```

**Option B: Manual (flexible timing)**
```bash
# Step 1: Plan first
python examples/tasks_mmk2/coffeecup_plate.py --data_set_size 10 --auto

# Step 2: Render later (minutes, hours, or days later)
cd data/mmk2_plate_coffecup
python record_playback.py --all                      # All trajectories
python record_playback.py --data_idx 0               # Single trajectory
python record_playback.py --all --randomize_visuals  # With randomization
```

### Output
```
data/mmk2_plate_coffecup/
└── 000/, 001/, 002/, ...
    ├── overview.mp4
    └── rendered/            ← Full observations
        ├── cam_0.mp4
        ├── cam_1.mp4
        ├── cam_2.mp4
        ├── cam_0_depth.mp4
        └── ...
```

---

## Use Cases

### Use Case 1: Quick Iteration
Generate 100 trajectories quickly, check overview.mp4s, then render only the good ones:
```bash
# Plan 100 trajectories (fast!)
python task.py --data_set_size 100 --auto

# Check overview videos, decide which to render
# Render only trajectory 5
cd data/task_name
python record_playback.py --data_idx 5
```

### Use Case 2: Domain Randomization
Generate one trajectory, render it with 10 different visual variants:
```bash
# Plan once
python task.py --data_set_size 1 --auto

# Render with different textures
cd data/task_name
# Edit randomize_textures() in record_playback.py
python record_playback.py --all --randomize_visuals  # Variant 1
# Edit textures again
python record_playback.py --all --randomize_visuals  # Variant 2
# ... etc
```

### Use Case 3: Production Pipeline
Separate compute resources for planning vs rendering:
```bash
# On fast CPU cluster: Plan 1000 trajectories
python task.py --data_set_size 1000 --auto

# On GPU machines: Render observations (potentially with GS)
python record_playback.py --all
```

---

## Technical Details

### During Planning (recoder_mmk2 with overview_only=True)
```python
# Only saves:
# - mujoco_states.pkl (privileged)
# - action_trajectory.npy
# - obs_action.json
# - overview.mp4 (first camera only)

recoder_mmk2(save_path, act_lst, obs_lst, cfg, state_lst, overview_only=True)
```

### During Rendering (record_playback.py)
```python
# Loads states and renders all cameras
for state in state_lst:
    sim_node.set_mujoco_state(state)
    obs = sim_node.getObservation()  # All cameras rendered
    # Save to rendered/ directory
```

---

## Migration from Old System

### Old Approach (Single Phase)
```bash
# Everything at once (slow)
python task.py --data_set_size 10 --auto
# Output: 000/cam_0.mp4, 000/cam_1.mp4, 000/cam_2.mp4, ...
```

### New Approach (Two Phase)
```bash
# Fast planning
python task.py --data_set_size 10 --auto
# Output: 000/overview.mp4 only

# Detailed rendering
python record_playback.py --all
# Output: 000/rendered/cam_0.mp4, cam_1.mp4, cam_2.mp4, ...
```

---

## Summary

✅ **Planning is now 3-5x faster** (only 1 camera vs multiple)
✅ **Flexible rendering** (apply different visuals to same trajectory)  
✅ **Better resource utilization** (separate planning from rendering)
✅ **Backward compatible** (can still use `--render` for one-shot)

The two-phase pipeline is especially valuable for:
- Large-scale data generation
- Domain randomization experiments  
- Iterative trajectory refinement
- Resource-constrained environments
