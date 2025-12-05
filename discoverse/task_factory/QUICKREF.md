# YAML Task Factory - Quick Reference

## File Structure

```
discoverse/
├── task_factory/
│   ├── task_factory.py          # Main factory
│   ├── motion_primitives.py     # Motion primitives
│   ├── yaml_utils.py           # Validation tools
│   └── README.md               # Full documentation
├── configs/tasks/
│   ├── cover_cup_yaml.yaml     # Example 1
│   └── pick_place_block.yaml   # Example 2
└── examples/tasks_airbot_play/
    ├── run_yaml_task.py        # Run script
    └── demo_yaml_factory.py    # Demo script
```

## Quick Start

```bash
# Validate a configuration
python -m discoverse.task_factory.yaml_utils config.yaml --validate

# Summarize a task
python -m discoverse.task_factory.yaml_utils config.yaml --summarize

# Run a task
python examples/tasks_airbot_play/run_yaml_task.py --config config.yaml

# Run demos
python examples/tasks_airbot_play/demo_yaml_factory.py
```

## YAML Structure

```yaml
# Required fields
task_name: "my_task"
robot_name: "airbot_play"
objects: [...]

robot_config:
  init_qpos: [...]
  simulation: {...}
  rendering: {...}

action_sequence:
  - type: "primitive_name"
    params: {...}

success_conditions:
  position_conditions: [...]
  orientation_conditions: [...]

# Optional fields
description: "..."
randomization: {...}
max_time: 20.0
move_speed: 0.75
```

## Motion Primitives Cheat Sheet

| Primitive | Purpose | Key Parameters |
|-----------|---------|----------------|
| `move_to_object` | Move to object | `object`, `offset`, `rotation`, `gripper` |
| `track_object_offset` | Move with local offset | `object`, `local_offset`, `rotation`, `gripper` |
| `move_to_position` | Move to absolute pos | `position`, `relative`, `rotation`, `gripper` |
| `offset_current` | Relative movement | `offset`, `gripper` |
| `grasp` | Close gripper | `position` (default: 0.0) |
| `release` | Open gripper | `position` (default: 0.04) |
| `delay` | Wait | `duration` |

## Success Conditions Cheat Sheet

| Type | Checks | Parameters |
|------|--------|------------|
| `object_near_object` | Distance between objects | `object1`, `object2`, `threshold` |
| `object_at_position` | Object at position | `object`, `position`, `threshold` |
| `planar_distance` | Distance in plane | `object1`, `object2`, `axes`, `threshold` |
| `object_upright` | Orientation check | `object`, `axis`, `threshold` |
| Custom expression | Custom Python | `expression` |

## Randomization Options

```yaml
randomization:
  # Object positions
  object_positions:
    - object: "obj_name"
      position_range: {x: 0.05, y: 0.05, z: 0.02}
  
  # Table height
  table_height: true
  table_config:
    table_name: "table"
    affected_objects: [...]
  
  # Materials
  materials: ["texture1", "texture2"]
  
  # Lighting
  lighting: true
  light_config: {direction: true, color: true, active: true}
  
  # Table texture
  table_texture: true
```

## Common Patterns

### Pick and Place
```yaml
action_sequence:
  - type: "move_to_object"          # Approach
  - type: "grasp"                   # Grasp
  - type: "delay"                   # Stabilize
  - type: "offset_current"          # Lift
  - type: "move_to_object"          # Move to target
  - type: "release"                 # Release
```

### Pour/Tilt
```yaml
- type: "move_to_object"
  params:
    object: "cup"
    rotation:
      euler: [0, 45, 0]             # Tilt
```

### Precise Placement
```yaml
- type: "track_object_offset"
  params:
    object: "target"
    local_offset: [0.0, 0.0, 0.1]   # Use local frame
```

## Tips

1. **Always validate first**: `--validate` before running
2. **Start simple**: Use `pick_place_block.yaml` as template
3. **Use delays**: After grasp/release for stability
4. **Offsets are key**: Approach safely before grasping
5. **Local vs world**: Use `track_object_offset` for rotating objects
6. **Test incrementally**: Add one action at a time
7. **Success thresholds**: Not too strict, not too loose

## Command Line Args

```bash
python run_yaml_task.py --config PATH [OPTIONS]

Options:
  --data_idx N          Start index (default: 0)
  --data_set_size N     Number of samples (default: 1)
  --auto                Headless mode
  --use_gs              Use Gaussian Splatting
```

## Python API

```python
from discoverse.task_factory import load_task_from_yaml

# Load task
TaskClass, cfg, factory = load_task_from_yaml("config.yaml")

# Create instance
sim_node = TaskClass(cfg)

# Get state executor
from discoverse.robots import AirbotPlayIK
arm_ik = AirbotPlayIK()
execute_state = factory.create_state_machine_executor(sim_node, arm_ik)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| IK Error | Reduce offsets, check reachability |
| Import Error | Run from DISCOVERSE root directory |
| Validation Error | Check YAML indentation and field names |
| Task Always Fails | Adjust success condition thresholds |
| Objects Wrong | Verify object names match MJCF |

## See Also

- Full docs: `discoverse/task_factory/README.md`
- Examples: `discoverse/configs/tasks/*.yaml`
- Demos: `python demo_yaml_factory.py`
