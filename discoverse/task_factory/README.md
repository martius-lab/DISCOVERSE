# YAML Task Factory

A powerful system for defining robotic manipulation tasks using YAML configuration files. This factory generates complete task classes with robot configuration, domain randomization, state machine execution, and success checking - all configured through YAML.

## Overview

The YAML Task Factory allows you to define complex manipulation tasks without writing Python code. Instead, you specify:

1. **Task metadata** (name, description, robot type)
2. **Robot configuration** (models, initial pose, simulation settings)
3. **Domain randomization** (object positions, materials, lighting)
4. **Action sequence** (motion primitives with parameters)
5. **Success conditions** (how to determine task completion)

## Quick Start

### 1. Create a YAML Configuration File

```yaml
task_name: "pick_and_place"
robot_name: "airbot_play"
description: "Pick up a block and place it on a target"

objects:
  - "red_block"
  - "target_plate"

robot_config:
  init_qpos: [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599, 0.0]
  # ... more config ...

action_sequence:
  - type: "move_to_object"
    params:
      object: "red_block"
      offset: [0, 0, 0.15]
      gripper: 0.04
  
  - type: "grasp"
    params:
      position: 0.0
  
  # ... more actions ...

success_conditions:
  position_conditions:
    - type: "object_near_object"
      object1: "red_block"
      object2: "target_plate"
      threshold: 0.05
```

### 2. Run the Task

```bash
# Run with GUI
python examples/tasks_airbot_play/run_yaml_task.py --config discoverse/configs/tasks/pick_place_block.yaml

# Run in automatic mode (headless)
python examples/tasks_airbot_play/run_yaml_task.py --config discoverse/configs/tasks/pick_place_block.yaml --auto --data_set_size 100

# Run with Gaussian Splatting renderer
python examples/tasks_airbot_play/run_yaml_task.py --config discoverse/configs/tasks/cover_cup_yaml.yaml --use_gs
```

### 3. Validate Your Configuration

```bash
python -m discoverse.task_factory.yaml_utils discoverse/configs/tasks/cover_cup_yaml.yaml --validate
python -m discoverse.task_factory.yaml_utils discoverse/configs/tasks/cover_cup_yaml.yaml --summarize
```

## Motion Primitives

Motion primitives are reusable action building blocks. Each primitive uses `get_body_tmat` to track object poses with optional offsets and rotations.

### Available Primitives

#### 1. `move_to_object`
Move end-effector to an object with optional offset and rotation.

```yaml
- type: "move_to_object"
  params:
    object: "red_cup"           # Object body name
    offset: [0, 0, 0.15]        # [x, y, z] offset in meters
    rotation:                    # Optional rotation
      euler: [0, 180, 0]
      seq: "xyz"
      degrees: true
    gripper: 0.04               # Gripper position (0-0.04)
```

#### 2. `track_object_offset`
Move to an object with offset in the object's local frame.

```yaml
- type: "track_object_offset"
  params:
    object: "cup"
    local_offset: [0.0, 0.06, 0.05]  # Offset in object's frame
    rotation:
      euler: [0, 72, 0]
      seq: "xyz"
      degrees: true
    gripper: 0.04
```

#### 3. `move_to_position`
Move end-effector to an absolute or relative position.

```yaml
- type: "move_to_position"
  params:
    position: [0.5, 0.2, 0.3]   # Target position [x, y, z]
    relative: false              # If true, relative to current
    rotation:
      euler: [0, 180, 0]
      seq: "xyz"
      degrees: true
    gripper: 0.04
```

#### 4. `offset_current`
Move relative to the current end-effector pose.

```yaml
- type: "offset_current"
  params:
    offset: [0, 0, 0.15]        # Move up 15cm
    gripper: 0.04               # Optional
```

#### 5. `grasp`
Close gripper to grasp an object.

```yaml
- type: "grasp"
  params:
    position: 0.0               # Gripper close position
```

#### 6. `release`
Open gripper to release an object.

```yaml
- type: "release"
  params:
    position: 0.04              # Gripper open position
```

#### 7. `delay`
Wait for a specified duration.

```yaml
- type: "delay"
  params:
    duration: 0.5               # Wait 0.5 seconds
```

## Success Conditions

Define conditions to determine if the task was completed successfully.

### Position Conditions

#### `object_near_object`
Check if two objects are within a threshold distance.

```yaml
position_conditions:
  - type: "object_near_object"
    object1: "block"
    object2: "target"
    threshold: 0.05             # 5cm
```

#### `object_at_position`
Check if an object is at a specific position.

```yaml
position_conditions:
  - type: "object_at_position"
    object: "block"
    position: [0.5, 0.2, 0.1]
    threshold: 0.03
```

#### `planar_distance`
Check distance in a specific plane (xy, xz, or yz).

```yaml
position_conditions:
  - type: "planar_distance"
    object1: "cup"
    object2: "plate"
    axes: "xy"                  # Check only x-y plane
    threshold: 0.02
```

### Orientation Conditions

#### `object_upright`
Check if an object maintains a specific orientation.

```yaml
orientation_conditions:
  - type: "object_upright"
    object: "cup"
    axis: "z"                   # Check z-axis is vertical
    threshold: 0.99             # Dot product threshold
```

### Custom Conditions

For complex conditions, use Python expressions.

```yaml
custom_conditions:
  - expression: "get_body_tmat(mj_data, 'cup')[2,3] > 0.15"
```

**Available context variables:**
- `np`: NumPy module
- `get_body_tmat`: Function to get object transformation
- `mj_data`: MuJoCo data object
- `abs`: Absolute value function

## Domain Randomization

Configure domain randomization to generate diverse training data.

### Object Position Randomization

```yaml
randomization:
  object_positions:
    - object: "red_block"
      position_range:
        x: 0.08               # +/- 8cm in x
        y: 0.08               # +/- 8cm in y
        z: 0.02               # +/- 2cm in z
```

### Table Height Randomization

```yaml
randomization:
  table_height: true
  table_config:
    table_name: "table"
    affected_objects:
      - "red_block"
      - "target_plate"
```

### Material Randomization

```yaml
randomization:
  materials:
    - "block_texture"
    - "plate_texture"
    - "cup_texture"
```

### Lighting Randomization

```yaml
randomization:
  lighting: true
  light_config:
    direction: true
    color: true
    active: true
```

### Table Texture Randomization

```yaml
randomization:
  table_texture: true
```

## Robot Configuration

### Gaussian Splatting Models

```yaml
robot_config:
  gs_models:
    background: "scene/lab3/point_cloud.ply"
    red_block: "object/block_red.ply"
    target_plate: "object/plate_white.ply"
```

### Initial Joint Positions

```yaml
robot_config:
  init_qpos: [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599, 0.0]
```

### Simulation Parameters

```yaml
robot_config:
  simulation:
    timestep: 0.004166667       # 1/240 seconds
    decimation: 4
    sync: true
    headless: false
```

### Rendering Parameters

```yaml
robot_config:
  rendering:
    fps: 20
    width: 640
    height: 480
  
  camera_ids: [0, 1]            # Cameras for observation
```

## Advanced Usage

### Using the Factory in Python

```python
from discoverse.task_factory import load_task_from_yaml, YAMLTaskFactory

# Load and run a task
TaskClass, cfg, factory = load_task_from_yaml("path/to/config.yaml")
sim_node = TaskClass(cfg)

# Or use the factory directly
factory = YAMLTaskFactory("path/to/config.yaml")
cfg = factory.setup_environment()
TaskClass = factory.create_task_class()
```

### Creating Custom Motion Primitives

```python
from discoverse.task_factory.motion_primitives import MotionPrimitive, MOTION_PRIMITIVES

class MyCustomPrimitive(MotionPrimitive):
    def execute(self, sim_node, arm_ik, tmat_armbase_2_world):
        # Your custom logic here
        return target_control

# Register your primitive
MOTION_PRIMITIVES['my_custom'] = MyCustomPrimitive
```

Then use it in YAML:

```yaml
action_sequence:
  - type: "my_custom"
    params:
      custom_param: value
```

### Validation and Debugging

```python
from discoverse.task_factory import validate_yaml_config, summarize_task

# Validate configuration
is_valid = validate_yaml_config("path/to/config.yaml")

# Print task summary
summarize_task("path/to/config.yaml")
```

## Examples

The repository includes several example configurations:

1. **`cover_cup_yaml.yaml`** - Complex multi-step task: pick cup, place on plate, cover with lid
2. **`pick_place_block.yaml`** - Simple pick and place task

See `discoverse/configs/tasks/` for more examples.

## File Structure

```
discoverse/
├── task_factory/
│   ├── __init__.py              # Module exports
│   ├── task_factory.py          # Main factory class
│   ├── motion_primitives.py     # Motion primitive definitions
│   └── yaml_utils.py            # Validation and utilities
├── configs/
│   └── tasks/
│       ├── cover_cup_yaml.yaml  # Example: cover cup task
│       └── pick_place_block.yaml # Example: pick and place
└── examples/
    └── tasks_airbot_play/
        └── run_yaml_task.py     # Run script
```

## Tips and Best Practices

1. **Start Simple**: Begin with a simple task (e.g., `pick_place_block.yaml`) and gradually add complexity.

2. **Validate First**: Always validate your YAML before running:
   ```bash
   python -m discoverse.task_factory.yaml_utils your_task.yaml --validate
   ```

3. **Use Offsets**: Use `offset` parameters to approach objects safely before grasping.

4. **Add Delays**: Add small delays after grasp/release actions for stability.

5. **Test Incrementally**: Test each action primitive individually before combining them.

6. **Use Local Offsets**: For rotating objects, use `track_object_offset` with `local_offset` to maintain relative positioning.

7. **Check Success Conditions**: Make success conditions neither too strict nor too loose.

8. **Randomization Range**: Start with small randomization ranges and increase gradually.

## Troubleshooting

### IK Errors
If you get IK (Inverse Kinematics) errors:
- Check that target positions are reachable
- Reduce offset distances
- Ensure rotations are reasonable

### Task Fails Success Check
- Verify success condition thresholds
- Use `--headless false` to watch the task execution
- Check object names match exactly

### Configuration Errors
- Use the validator: `python -m discoverse.task_factory.yaml_utils config.yaml --validate`
- Check indentation (YAML is sensitive to indentation)
- Verify all required fields are present

## Future Extensions

This factory system can be extended to support:
- Other robot types (SO101, MMK2, etc.)
- Vision-based primitives
- Force-based primitives
- Trajectory recording and replay
- Multi-robot tasks
- Conditional execution (if-then logic)

## Contributing

To add new motion primitives:
1. Create a new class inheriting from `MotionPrimitive` in `motion_primitives.py`
2. Implement the `execute()` method
3. Register it in the `MOTION_PRIMITIVES` dictionary
4. Document the parameters and usage

## License

See the main DISCOVERSE LICENSE file.
