# DISCOVERSE Task Generation System Spec

This document captures a concrete, codebase‑aligned design for scaling procedural task generation in DISCOVERSE. It integrates a typed world model, symbolic operators, compositional “wrapper” patterns, a task spec language, feasibility/necessity checks, and reward shaping, while remaining compatible with the current MuJoCo scene assembly, YAML task configs, and runtime executor.


## 1. Overview

- Scene assembly: `discoverse/envs/make_env.py` merges a robot MJCF and a pure task environment MJCF into a composite model with a consistent keyframe.
- Task definition: YAML under `discoverse/configs/tasks/*.yaml` provides states (primitives), randomization, and success checks; templates under `discoverse/configs/tasks/templates/*`.
- Runtime: `examples/universal_tasks/universal_task_runtime.py` executes state sequences via IK (`MinkIKSolver`) and a simple state machine. `UniversalTaskBase` (`discoverse/universal_manipulation/task_base.py`) ties robot config, task config, and `SceneRandomizer` (`discoverse/universal_manipulation/randomization.py`).
- This spec adds: a world model (objects/tags), a predicate library, an operator extension (primitives), compositional obstruction “wrappers”, a task generator, and reward shaping — all serialized in extended YAML and implemented with small, surgical changes.


## 2. World Model: Objects, Tags, Predicates

### 2.1 Object Schema (YAML)

Per object instance (extending existing `objects` entries):

- `name`: MJCF body name (existing convention)
- `type`: `block`, `cylinder`, `mug`, `bottle`, `box`, `bin`, `tray`, `platform`, `drawer`, `door`, `knob`, `peg`, `hole`, `tool`, `target_region`, etc.
- `tags`: list of affordances
  - Core: `graspable`, `pushable`, `pullable`, `liftable`, `insertable(role=peg/hole, axis, tol)`
  - Container / support: `container(openable, opening_state, rim_height)`, `receptacle(inside_region, surface_region)`, `platform(support_area, margin_tolerance)`, `support_surface`
  - Articulation: `has_handle(handle_pose, approach_dir)`, `articulated(joint_type, limits, stiffness)`
  - Optional: `fragile`, `heavy`, `slippery`
- `articulation` (optional): `{ joint: <joint_name>, type: slide|hinge, limits: [min,max], handle_site: <site_name> }`
- `regions` (optional): precomputed regions relative to object (e.g., `{ inside: { body: <name>, type: cylinder|box, ... }, surface: {...} }`)

Notes:
- Geometry, size, and pose remain authoritative in MJCF; YAML only references them by names and additional metadata. Avoid duplicating meshes.
- For articulated entities (drawer/door), store the joint and handle `site` name for motion/IK targets.

### 2.2 Predicates (MuJoCo‑evaluated)

Geometric:
- `On(o1, o2)`
- `In(o1, o2)`
- `Near(o1, region)` / `At(o1, region)`
- `Aligned(o1, o2, axis, tol)`
- `Upright(o, tol)`
- `Inserted(peg, hole, depth, angle_tol)`
- `Blocks(blocker, target)` (occlusion/contact relation)
- `Reachable(o)` (IK + collision budget)
- `Visible(o)` (camera frustum + ray/occlusion)

Object-level:
- `Held(o)`
- `Clear(o)` (no object `On(o)`)
- `Open(x)` / `Closed(x)` (articulation state)
- `Free(o)` / `Anchored(o)` (workspace constraints)

Derived examples:
- `Access(o) := Reachable(o) ∧ Visible(o) ∧ Clear(o)`

Implementation plan:
- New module `discoverse/universal_manipulation/predicates.py` exporting functions that query MuJoCo state (`mj_data`, `mj_model`) to evaluate these relations. Use helpers from `discoverse/utils/__init__.py` (e.g., `get_body_tmat`, `get_site_tmat`).
- Integrate into `UniversalTaskBase._evaluate_condition` (`discoverse/universal_manipulation/task_base.py`) by adding new `condition.type` branches (non‑breaking; existing types like `distance_2d`, `orientation` remain).


## 3. Operator Library (Skill/Gesture Primitives)

Operator forms with `preconditions` (predicates+tags) and `effects`:

- `REACH(o)` → pre: `Reachable(o) ∧ Visible(o)`, eff: `Near(gripper, o)`
- `GRASP(o, grasp_site)` → pre: `Near(gripper, o) ∧ graspable(o) ∧ Clear(o)`, eff: `Held(o)`
- `LIFT(o, h)` → pre: `Held(o)`, eff: `Height(o) ≥ h`
- `PLACE_ON(o, s)` → pre: `Held(o) ∧ support_surface(s) ∧ Reachable(s)`, eff: `On(o, s) ∧ ¬Held(o)`
- `PLACE_IN(o, c)` → pre: `Held(o) ∧ container(c) ∧ Reachable(c)`, eff: `In(o, c) ∧ ¬Held(o)`
- `PUSH(o, dir, dist)` → pre: `pushable(o) ∧ Reachable(o)`, eff: pose shifted
- `PULL(o, dir, dist)` → pre: `pullable(o) ∧ Reachable(o)`, eff: pose shifted
- `OPEN_JOINT(x, θ)` → pre: `articulated(x) ∧ has_handle(x) ∧ Reachable(handle(x))`, eff: `Open(x, ≥θ)`
- `CLOSE_JOINT(x)` → eff: `Closed(x)`
- `INSERT(peg, hole)` → pre: `Held(peg) ∧ insertable(peg/hole) ∧ Aligned(peg, hole) ∧ Reachable(hole)`, eff: `Inserted(peg, hole)`
- `UNBLOCK(target, blocker)` → pre: `Blocks(blocker, target) ∧ pushable(blocker) ∧ Reachable(blocker)`, eff: `¬Blocks(blocker, target)`

Integration as primitives:
- Extend `examples/universal_tasks/universal_task_runtime.py` in `set_target_from_primitive` to support: `reach`, `lift`, `place_on`, `place_in`, `push`, `pull`, `open_joint`, `close_joint`, `insert`, `unblock`.
- Preconditions can be optionally checked via the predicate layer to early‑fail a state if violated (demo mode remains permissive).


## 4. Task Spec Language (TSL)

A TSL task consists of:
- `goal`: logical formula over predicates (string or structured), optionally with ordering
- `base_goal_family`: e.g., `pick_and_place`, `insertion`, `open_then_place` (label for generator)
- `obstructions`: list of wrapper patterns applied to base goal
- `objects`: concrete instances & tags (see 2.1)
- `expected_min_steps`: symbolic minimal steps
- `subgoals`: sequence derived from causal plan (for reward shaping)

YAML integration:
- Existing YAML is extended with optional fields: `goal`, `base_goal_family`, `obstructions`, expanded `objects` metadata.
- `TaskConfigLoader` (`discoverse/universal_manipulation/task_config.py`) accepts these keys. For `goal`, compile into the current `success_check` format when possible (e.g., `On` → distance + height + orientation thresholds; `In` → point‑in‑region).


## 5. Obstruction Patterns as Compositional Wrappers

Start from base goal families (minimal plans) and apply wrapper patterns that modify initial state, add objects, and force additional steps.

Base goal families:
- `Grasp(X)` → goal: `Held(X)`
- `PutIn(X, C)` → goal: `In(X, C)`
- `PutOn(X, S)` → goal: `On(X, S)`
- `Open(D)` → goal: `Open(D)`
- `Insert(Peg, Hole)` → goal: `Inserted(Peg, Hole)`
- `Stack(X, Y)` → goal: `On(X, Y) ∧ Upright(X)`

Wrapper patterns:
1) Out‑of‑reach‑with‑pull
   - Introduce container `C` (`pullable`); set `In(X, C)`; place `C` where `Reachable(C) ∧ ¬Reachable(X)`; require `PULL(C, …)` before `GRASP(X)`.

2) Closed‑receptacle
   - Introduce `D` (`container(openable)`) with `In(X, D)` and `Closed(D)`; require `OPEN_JOINT(D)` before `GRASP(X)`.

3) Occluded‑object
   - Introduce blocker `B` (`pushable`); set `On(B, X)` and `Blocks(B, X)`; require `UNBLOCK` (e.g., `PUSH(B)`).

4) Barrier/gap
   - Add wall/constraint geometry; a straight path collides, requiring intermediate maneuver/reorientation.

5) Platform‑then‑place
   - Introduce platform `P`; require `PLACE_ON(X, P)` then regrasp and `PLACE_ON(X, tiny_target)`.

6) Precision‑insertion
   - Tight alignment/angle tolerances; require reorientation/platform staging before `INSERT`.

Practical note:
- Use existing MJCFs like `models/mjcf/task_environments/open_drawer.xml`, `peg_in_hole.xml`, `block_bridge_place.xml` as starting points. For occluders/platforms/walls, add them once to appropriate environment MJCFs and activate them via placement/randomization and wrappers.


## 6. Task Generation Algorithm

Per episode:
1) Sample base goal family (compatible with available objects/tags).
2) Instantiate objects and initial poses (use `randomization.objects` with `min_distance`; collision checks handled by MuJoCo forward + optional contacts).
3) Sample obstruction patterns compatible with the base goal and apply wrappers:
   - Modify initial predicates (e.g., `Closed(D)`, `Blocks(B, X)`).
   - Add required objects (select from MJCF inventory) and place them.
   - Extend the `states` sequence with the induced operators.
4) Compile symbolic domain/problem
   - Domain: operator library; Problem: initial predicates + `goal`.
   - For v1, the “plan” is the state sequence we assembled; an external planner can be integrated later.
5) Necessity (skip‑step) checks
   - Remove wrapper‑induced state(s) and simulate if the `goal` is still reachable; if yes, reject and resample.
   - Or clamp predicate (e.g., keep drawer `Closed`) and verify `goal` unattainable.
6) Geometric feasibility
   - For key states (grasp, place, insert), run IK (`MinkIKSolver`) and quick MuJoCo step checks (contacts/penetration); reject if infeasible.
7) Subgoal automaton
   - Extract subgoals as predicates made newly true after each operator (e.g., `Open(D)`, `Held(X)`, `In(X,C)`); store as `subgoals` for reward shaping.
8) Emit final TaskSpec
   - YAML with extended sections (`objects`, `goal`, `obstructions`, `randomization`, `states`, `success_check`, `expected_min_steps`, `subgoals`). Saved under `discoverse/configs/tasks/generated/`.

Implementation module:
- `discoverse/universal_manipulation/generator.py` providing APIs to sample a base goal, apply wrappers, validate, and save YAML.


## 7. Reward Generation

Use potential‑based shaping over subgoals:
- Success: when all goal predicates true for a short hold.
- Subgoal potentials `φ_i(s) ∈ [0,1]` for reach/grasp/lift/place/open/unblock/insert, e.g.:
  - Reach: distance‑based proximity to target/handle.
  - Grasp: binary `Held(o)` or contact‑based heuristic.
  - Lift: normalized height progress.
  - Place: normalized distance to target region + orientation.
  - Open: normalized joint angle.
  - Insert: normalized depth/angle error.
- Per step: `r_t = Φ(s_{t+1}) − Φ(s_t) + bonuses_on_completion − step_penalty`.

Module:
- `discoverse/universal_manipulation/reward.py` with `RewardShaper` and plug‑in to `UniversalRuntimeTaskExecutor` (optional flag). Executor file: `examples/universal_tasks/universal_task_runtime.py`.


## 8. Integration Points (Minimal Changes)

- New: `discoverse/universal_manipulation/predicates.py` (predicate evaluators).
- Update: `discoverse/universal_manipulation/task_base.py`
  - Extend `_evaluate_condition` to support `on`, `in`, `upright`, `held`, `open`, `closed`, `reachable`, `visible`.
- Update: `examples/universal_tasks/universal_task_runtime.py`
  - Extend `set_target_from_primitive` with new primitives (`reach`, `lift`, `place_on`, `place_in`, `push`, `pull`, `open_joint`, `close_joint`, `insert`, `unblock`).
- Update: `discoverse/universal_manipulation/randomization.py`
  - Optional: add `articulations` initialization (set joint qpos ranges/states), and wrapper helpers to place occluders/platforms.
- New: `discoverse/universal_manipulation/generator.py` (wrappers, resampling loop, YAML emit).
- New: `discoverse/universal_manipulation/reward.py` (optional shaping).
- Update: `discoverse/universal_manipulation/task_config.py`
  - Non‑breaking: accept/access new fields (`goal`, `base_goal_family`, `obstructions`, extended `objects`).


## 9. Milestones

1) Predicate layer + condition support
   - Implement `On/In/Upright/Held/Open/Closed/Reachable/Visible`.
2) Primitive extensions
   - Implement `reach/lift/place_on/place_in/open_joint/close_joint`; simple `push/pull`.
3) Generator + wrappers
   - Implement wrappers: `closed_receptacle`, `occluded_object`, `out_of_reach_with_pull`; emit to `configs/tasks/generated/` with MuJoCo feasibility + skip‑step necessity checks.
4) Reward shaping
   - Add `RewardShaper`; log potentials and shaped rewards.
5) Articulation randomization
   - Initialize drawer/door joints; randomize within ranges for diversity.


## 10. Concrete Example YAML (Generated)

```yaml
task_name: "putin_block_closed_drawer"
base_goal_family: "put_in"
description: "Open drawer to access the block, then place into bowl"

objects:
  - name: "block_green"
    type: "block"
    tags: ["graspable", "liftable"]
  - name: "bowl_pink"
    type: "container"
    tags: ["container", "receptacle"]
    regions:
      inside: { body: "bowl_pink", type: "cylinder", radius: 0.055, height: 0.04 }
  - name: "drawer_1"
    type: "drawer"
    tags: ["container", "openable", "has_handle"]
    articulation: { joint: "drawer_1_slide", type: "slide", limits: [0.0, 0.25], handle_site: "drawer_1_handle_site" }

goal: 'In(block_green, bowl_pink) ∧ Upright(block_green, 0.95)'
expected_min_steps: 4

success_check:
  method: "combined"
  operator: "and"
  conditions:
    - { type: "in", object: "block_green", container: "bowl_pink" }
    - { type: "orientation", object: "block_green", axis: "z", direction: "up", threshold: 0.95 }

randomization:
  objects:
    - { name: "block_green", x_range: [0.1, 0.25], y_range: [-0.2, 0.2], min_distance: 0.05 }
  articulations:
    - { joint: "drawer_1_slide", qpos_range: [0.0, 0.02] }

states:
  - { name: "reach_handle", primitive: "reach", params: { object_name: "drawer_1", site: "drawer_1_handle_site" }, gripper_state: "open" }
  - { name: "open_drawer", primitive: "open_joint", params: { joint_name: "drawer_1_slide", target: 0.20 }, gripper_state: "open" }
  - { name: "approach_block", primitive: "move_to_object", params: { object_name: "block_green", offset: [0, 0, 0.005] }, gripper_state: "open" }
  - { name: "grasp_block", primitive: "grasp_object", params: { object_name: "block_green" }, gripper_state: "close", delay: 0.3 }
  - { name: "lift_block", primitive: "lift", params: { height: 0.07 }, gripper_state: "close" }
  - { name: "place_in_bowl", primitive: "place_in", params: { container: "bowl_pink", height: 0.05 }, gripper_state: "close" }
  - { name: "release_block", primitive: "release_object", params: { object_name: "block_green" }, gripper_state: "open" }
```

This remains compatible with the existing executor and success checks while introducing semantic richness.


## 11. Implementation Notes & Conventions

- Prefer `min_distance` for object spacing (the current randomizer uses it). Standardize away from `collision_radius`.
- Keep YAML extensions optional for backward compatibility. Existing tasks continue to run without the new fields.
- For new predicates, start with robust approximations (distance/height thresholds) before adding contact/ray‑based precision.
- For obstruction necessity, use simulation‑based skip‑step checks first (lightweight) before integrating external symbolic planners.


## 12. Future Work

- External planner integration for minimal plan proofs and counterfactual checks.
- Tool‑use and multi‑object constraints; multi‑goal conjunctions.
- Richer visibility/occlusion via renderer buffers or ray casting.
- Automatic MJCF augmentation (adding occluders/walls/platforms programmatically) if needed.


## Test intents
  - When constructing a minimal task YAML for task name "place_block" (matching the existing pure task MJCF) under the tasks config directory and invoking robot–task model generation for robot "airbot_play" and task "place_block", it should produce the merged MJCF file at `.../mjcf/tmp/airbot_play_place_block.xml` that MuJoCo can load and render at least one frame without error.
  - When loading a legacy task YAML that lacks goal, objects metadata, or obstructions, it should execute with its original states and success checks unchanged and complete successfully under the same
    initial conditions.
  - When loading a task YAML that specifies a goal “On(block_green, block_blue)” and the scene is prepared with the green block centered over the blue block within 0.025m 2D distance and correct stack
    height, it should report success for the goal.
  - When loading a task YAML that specifies a goal “In(block_green, bowl_pink)” and the green block is placed within the defined inside region of the bowl, it should report success for the goal.
  - When loading a task YAML that specifies a goal “Upright(block_green, 0.99)” and the block’s local z-axis aligns with world z with dot product exactly 0.99, it should report success for the goal.
  - When loading a task YAML that specifies a goal “Upright(block_green, 0.99)” and the block’s dot product to world z is 0.989, it should report failure for the goal.
  - When a scene is configured with a closed drawer (slide joint at 0.0) containing a block and the goal is “Open(drawer_1)”, it should report success only after the drawer joint reaches or exceeds the
    specified open target.
  - When a scene has a held peg aligned with a hole within a 5-degree angular tolerance and insertion depth threshold reached, it should report success for the goal “Inserted(peg_1, hole_1)” and
    failure if the angle error is 5.1 degrees with the same depth.
  - When a blocker object sits on top of a target object and the goal is “¬Blocks(blocker, target)”, it should report failure before the blocker is moved away and success after the blocker is moved to
    a clear region.
  - When a target object lies beyond the robot’s reachable workspace, “Reachable(target)” should be false; after pulling its support into the workspace, “Reachable(target)” should be true.
  - When an object lies within the active camera’s view frustum and not occluded by intervening geometry, “Visible(object)” should be true; when a wall is placed fully blocking line-of-sight,
    “Visible(object)” should be false.
  - When the gripper closes on a graspable object in contact with the intended grasp region, “Held(object)” should become true and should return to false after a release action.
  - When another object is placed on top of a support object, “Clear(support)” should be false, and after removing the object “Clear(support)” should be true.
  - When a table is defined as anchored and a loose object is defined as free, “Anchored(table)” should be true and “Free(object)” should be true at reset.
  - When “Access(object)” is defined as the conjunction of “Reachable ∧ Visible ∧ Clear” and any one of those three is false, “Access(object)” should be false; when all three are true, “Access(object)”
    should be true.
  - When executing a “reach” step to a specified handle site, the end effector pose should end within 0.01m of the target site and the state should complete.
  - When attempting a “grasp” step without being near the object (more than 0.05m away), it should not mark the object as held and the state should not be considered complete.
  - When executing a “grasp” step while near a graspable object, the object should become held and the state should complete.
  - When executing a “lift” step with a held object and target height h=0.07m relative to the pick location, the object’s height should reach at least 0.07m and the state should complete.
  - When executing a “place_on” step while holding an object above a support surface, the object should end placed on the support (On true) and no longer held upon completion.
  - When executing a “place_in” step while holding an object above a container’s inside region, the object should end inside the container (In true) and no longer held upon completion.
  - When executing a “push” step with direction +X and distance 0.10m on a pushable object, the object’s final pose should shift by 0.10m along +X within a 0.01m tolerance and the state should
    complete.
  - When executing a “pull” step with direction −X and distance 0.10m on a pullable object, the object’s final pose should shift by 0.10m along −X within a 0.01m tolerance and the state should
    complete.
  - When executing an “open_joint” step to a target value (e.g., 0.20 for a slide joint) with the handle reachable, the joint position should reach at least the target and the container should be
    considered open.
  - When executing a “close_joint” step on an open jointed object, the joint position should return to the closed threshold and the container should be considered closed.
  - When executing an “insert” step with a held peg and a hole, if the peg is aligned within the specified angular tolerance and the insertion depth is reached, “Inserted” should become true; otherwise
    it should remain false and the state should fail to complete.
  - When executing an “unblock” step where a blocker overlaps a target, after the action the overlap should cease and “Blocks(blocker, target)” should be false.
  - When applying object randomization with non-overlapping x/y ranges and min_distance=0.05 for two objects, their sampled positions should satisfy the min_distance constraint in the final placements.
  - When applying object randomization where the bounds make non-overlapping placement impossible and max_attempts=100, it should fail the randomization after 100 attempts without changing object
    positions.
  - When applying camera randomization with position_offset [0.05, 0.05, 0.05] and orientation_offset [0.05, 0.05, 0.05], the camera’s position and orientation should change within those bounds
    relative to the initial pose.
  - When applying lighting randomization with random_color, random_active, and intensity_range [0.1, 0.9], each light’s color and intensity should be within [0,1], at least one light should be active,
    and intensities should fall within [0.1, 0.9] after scaling.
  - When applying table_height randomization with height_range [0.0, 0.1] and affected_objects [A,B], the table’s Z should lower by a value within the range and the listed objects’ Z should be adjusted
    by the same amount.
  - When applying texture randomization to a named texture with a viewer and renderer active, the texture’s image data should change and subsequent renders should reflect the updated texture.
  - When applying texture randomization without a viewer or renderer set, it should complete without error and skip display updates.
  - When initializing articulations via randomization (e.g., a drawer slide joint with qpos_range [0.0, 0.02]), the joint position at reset should lie within the specified range.
  - When generating a task from the base goal family “PutIn(X, C)” without obstructions and providing compatible tagged objects, the emitted YAML should contain a minimal sequence “grasp then place_in”
    and succeed in execution.
  - When applying the “closed_receptacle” wrapper to a grasp goal, the emitted task should require an “open” step before a successful grasp and should fail to achieve the goal if the open step is
    skipped.
  - When applying the “out_of_reach_with_pull” wrapper to a grasp goal, the emitted task should require a pull step that moves the container before grasping, and should fail to achieve the goal without
    performing the pull.
  - When applying the “occluded_object” wrapper to a grasp goal, the emitted task should require moving the blocker away before grasping, and should fail to achieve the goal while the blocker remains.
  - When running skip-step necessity checks on a generated task with an “open” wrapper, removing the open step should prevent the goal from being achieved under the same initial conditions.
  - When running geometric feasibility checks on a generated insertion task, if the approach, alignment, or depth is infeasible for IK or causes collision, the generator should reject the task and
    resample.
  - When emitting a generated task YAML, it should be saved under the configured output directory, parse successfully by the existing loader, and execute end-to-end in the runtime.
  - When compiling a “goal” field with a logical “and” over two predicates, it should only report success when both predicates are satisfied; when compiling with a logical “or”, it should report
    success when either predicate is satisfied.
  - When building a subgoal sequence from a generated plan, each subgoal should correspond to the newly satisfied predicate(s) after each operator, and the observed episode should mark subgoals as
    achieved in the same order during successful execution.
  - When computing shaped rewards, approaching the next subgoal should increase the potential, and the per-step reward should equal the potential difference plus any configured completion bonus minus
    the step penalty.
  - When computing success rewards with a specified hold time H, it should only grant the success reward if the goal predicates remain true continuously for at least H seconds of simulation time.
  - When setting a fixed randomization seed in the task configuration and running reset twice, the sampled object and camera placements should be identical across the two runs; when the seed is null,
    the two runs should differ in at least one randomized placement.
  - When running an existing task that does not specify the extended objects metadata or goal field, it should complete successfully and produce identical states and results compared to the pre-
    extension baseline.
  - When a container is closed and a “place_in” step is attempted directly, the object should not end inside the container and the state should not complete until the container is opened first.
  - When testing the boundary for “In” with the object center positioned exactly at the region radius and within the height, it should be considered inside; when positioned 1 mm beyond the radius, it
    should be considered outside.