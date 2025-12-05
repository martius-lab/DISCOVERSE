# YAML Task Factory - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          YAML Configuration File                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ task_name: "cover_cup"                                            │  │
│  │ robot_name: "airbot_play"                                         │  │
│  │ objects: [...]                                                    │  │
│  │ robot_config: {...}                                               │  │
│  │ randomization: {...}                                              │  │
│  │ action_sequence:                                                  │  │
│  │   - type: "move_to_object"                                        │  │
│  │   - type: "grasp"                                                 │  │
│  │   - ...                                                           │  │
│  │ success_conditions: {...}                                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         YAMLTaskFactory                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ __init__(yaml_path)                                               │  │
│  │   • Parse YAML                                                    │  │
│  │   • Create motion primitives                                      │  │
│  │   • Validate configuration                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Robot Config     │  │ Motion Sequence  │  │ Success Checker  │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ • GS models      │  │ Primitive 0:     │  │ Position checks: │
│ • init_qpos      │  │  MoveToObject    │  │  • near_object   │
│ • timestep       │  │ Primitive 1:     │  │  • at_position   │
│ • camera IDs     │  │  Grasp           │  │                  │
│ • render params  │  │ Primitive 2:     │  │ Orientation:     │
└──────────────────┘  │  Delay           │  │  • upright       │
                      │ Primitive 3:     │  └──────────────────┘
                      │  OffsetCurrent   │
                      │ ...              │
                      └──────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Generated Task Class                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ class GeneratedTask(AirbotPlayTaskBase):                          │  │
│  │                                                                   │  │
│  │   def domain_randomization(self):                                │  │
│  │       # Generated from randomization config                      │  │
│  │       for obj in randomized_objects:                             │  │
│  │           randomize_position(obj)                                │  │
│  │       randomize_lighting()                                       │  │
│  │                                                                   │  │
│  │   def check_success(self):                                       │  │
│  │       # Generated from success_conditions                        │  │
│  │       return all_conditions_met                                  │  │
│  │                                                                   │  │
│  │   # State machine execution via factory.create_state_executor()  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Execution Loop                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 1. sim_node.reset()                                               │  │
│  │ 2. domain_randomization()                                         │  │
│  │ 3. For each state in state_machine:                               │  │
│  │      a. Execute motion primitive                                  │  │
│  │      b. Get target_control from primitive                         │  │
│  │      c. Step simulation                                           │  │
│  │      d. Record observations                                       │  │
│  │ 4. check_success()                                                │  │
│  │ 5. Save data if successful                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Outputs                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Videos       │  │ Actions      │  │ Observations │                  │
│  │ cam_0.mp4    │  │ obs_action   │  │ Joint states │                  │
│  │ cam_1.mp4    │  │ .json        │  │ Timestamps   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘


Motion Primitive Execution Flow:
═════════════════════════════════

┌──────────────┐
│ YAML Action  │  "move_to_object: {object: 'cup', offset: [0,0,0.1]}"
└──────┬───────┘
       │
       ▼
┌───────────────────────┐
│ create_primitive()    │  Parse YAML → Create MoveToObjectPrimitive
└──────┬────────────────┘
       │
       ▼
┌───────────────────────────────────────────┐
│ MoveToObjectPrimitive.execute()           │
│  1. tmat = get_body_tmat(sim, "cup")      │  ← Track object pose
│  2. Apply offset: tmat[:3,3] += offset    │  ← Add offset
│  3. Apply rotation if specified           │
│  4. tmat_local = arm_base^-1 @ tmat       │  ← Transform to arm base
│  5. target_control[:6] = IK(pos, rot)     │  ← Solve IK
│  6. target_control[6] = gripper           │  ← Set gripper
│  7. return target_control                 │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│ State Machine                             │
│  • Set sim_node.target_control            │
│  • Interpolate to target                  │
│  • Check if action done                   │
│  • Move to next state                     │
└───────────────────────────────────────────┘


Success Checking Flow:
══════════════════════

YAML Success Conditions → Generated check_success() → Boolean Result
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
            │ Position     │ │ Orient.   │ │ Custom      │
            │ Checks       │ │ Checks    │ │ Expressions │
            │              │ │           │ │             │
            │ get_body_    │ │ Check     │ │ eval()      │
            │ tmat() →     │ │ rotation  │ │ in context  │
            │ compare      │ │ matrices  │ │             │
            └──────┬───────┘ └─────┬─────┘ └──────┬──────┘
                   │               │               │
                   └───────────────┴───────────────┘
                                   │
                                   ▼
                          All conditions AND'd
                                   │
                                   ▼
                          True/False result


Data Collection Pipeline:
═════════════════════════

┌──────────────┐
│ Run N Times  │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────┐
│ For each episode:                  │
│  1. Reset environment              │
│  2. Randomize (if enabled)         │
│  3. Execute action sequence        │
│  4. Record:                        │
│     • Video frames (PyAV encoder)  │
│     • Joint positions              │
│     • Actions                      │
│     • Timestamps                   │
│  5. Check success                  │
│  6. Save if successful             │
│  7. Increment counter              │
└────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│ Output Directory Structure:        │
│  data/task_name/                   │
│    000/                            │
│      cam_0.mp4                     │
│      cam_1.mp4                     │
│      obs_action.json               │
│    001/                            │
│      ...                           │
│    task.mjb                        │
│    task_config.yaml                │
└────────────────────────────────────┘
```

## Component Interactions

```
User → YAML File → Factory → Task Class → Simulation → Data

Tools:
  • Validator: YAML → Errors/Warnings
  • Summarizer: YAML → Human-readable text  
  • Demo: Show all components in action
```

## Extension Points

1. **Add Motion Primitive**: Subclass `MotionPrimitive` + register
2. **Add Success Condition**: Modify `check_success()` generator
3. **Add Robot Type**: Create config class + extend factory
4. **Add Randomization**: Extend `domain_randomization()` generator

## Key Design Principles

- **Separation of Concerns**: Config, primitives, execution separated
- **Composability**: Primitives combine to form complex behaviors  
- **Extensibility**: Easy to add new primitives and conditions
- **Validation**: Catch errors early with validator
- **Compatibility**: Generated tasks work with existing code
