"""
Prototype task generator implementing the wrapper patterns described in spec.md.

The focus of the spike is determinism and testability rather than exhaustive
random sampling.  The generator operates on top of the lightweight world-state
and primitive-controller utilities so it can reason about predicate feasibility
without relying on the full MuJoCo runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from .predicates import DictWorldState, ObjectMetadata, PredicateContext, PredicateEvaluator
from .primitives import PrimitiveController


@dataclass
class GeneratorConfig:
    task_name: str
    base_goal_family: str
    objects: List[Dict]
    wrappers: List[str] = field(default_factory=list)
    random_seed: Optional[int] = None


@dataclass
class GeneratedTask:
    task_name: str
    yaml_path: Path
    specification: Dict


class TaskGenerator:
    """Generates structured task specifications with optional wrappers."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or Path("discoverse") / "configs" / "tasks" / "generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def generate(self, config: GeneratorConfig) -> GeneratedTask:
        if config.random_seed is not None:
            np.random.seed(int(config.random_seed))

        world, metadata = self._initialise_world(config.objects)
        predicates = PredicateEvaluator(world, metadata, PredicateContext())
        controller = PrimitiveController(world, predicates)
        if config.objects:
            predicates.context.gripper_position = np.array(config.objects[0].get("initial_position", [0.0, 0.0, 0.04]), dtype=float)

        base_states, goal, success_check, subgoals = self._base_plan(config.base_goal_family, config.objects)
        wrappers_applied = []
        requested_wrappers = [wrapper.lower() for wrapper in config.wrappers]

        for wrapper in requested_wrappers:
            if wrapper == "closed_receptacle":
                addon_states = self._apply_closed_receptacle(controller, predicates, config.objects)
            elif wrapper == "out_of_reach_with_pull":
                addon_states = self._apply_out_of_reach(controller, predicates, config.objects)
            elif wrapper == "occluded_object":
                addon_states = self._apply_occluded_object(controller, predicates, config.objects)
            else:
                addon_states = []
            if addon_states:
                wrappers_applied.append(wrapper)
                base_states = addon_states + base_states

        # Validate necessity (rudimentary: ensure removing wrapper states fails)
        evaluation_world = world.clone()
        self._assert_necessity(
            controller,
            predicates,
            base_states,
            goal,
            wrappers_applied,
            evaluation_world,
            metadata,
            predicates.context,
        )

        if not self._plan_executes(world, metadata, predicates.context, base_states, goal):
            raise ValueError("Generated plan does not satisfy the goal.")

        subgoals = self._augment_subgoals(subgoals, requested_wrappers, config.objects)

        spec = self._build_spec_dict(
            config.task_name,
            config.base_goal_family,
            config.objects,
            base_states,
            goal,
            wrappers_applied,
            success_check,
            subgoals,
        )

        yaml_path = self.output_dir / f"{config.task_name}.yaml"
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, sort_keys=False)

        return GeneratedTask(config.task_name, yaml_path, spec)

    # ----------------------------------------------------------------- helpers
    def _initialise_world(self, objects: Iterable[Dict]) -> tuple[DictWorldState, Dict[str, ObjectMetadata]]:
        world = DictWorldState()
        metadata: Dict[str, ObjectMetadata] = {}
        for obj in objects:
            name = obj["name"]
            position = obj.get("initial_position", [0.0, 0.0, 0.0])
            quat = obj.get("initial_quat", [0.0, 0.0, 0.0, 1.0])
            geom_size = obj.get("geom_size", [0.02, 0.02, 0.02])
            world.add_body(name, position, quat, geom_size)

            if handle := obj.get("handle_site"):
                offset = obj.get("handle_offset", [0.0, 0.0, 0.0])
                world.add_site(handle, np.array(position) + np.array(offset))

            articulation = obj.get("articulation", {})
            joint_name = articulation.get("joint")
            limits = articulation.get("limits")
            handle_site = articulation.get("handle_site") or obj.get("handle_site")
            open_value = articulation.get("open")
            closed_value = articulation.get("closed")

            metadata[name] = ObjectMetadata(
                name=name,
                type=obj.get("type"),
                tags=set(obj.get("tags", [])),
                regions=obj.get("regions", {}),
                articulation_joint=joint_name,
                articulation_limits=tuple(limits) if limits else None,
                articulation_open_value=open_value,
                articulation_closed_value=closed_value,
                handle_site=handle_site,
            )

            if joint_name is not None:
                default_value = articulation.get("default")
                if default_value is not None:
                    world.set_joint(joint_name, default_value)
                elif open_value is not None:
                    world.set_joint(joint_name, open_value)
                elif closed_value is not None:
                    world.set_joint(joint_name, closed_value)
                elif limits:
                    world.set_joint(joint_name, limits[0])

        return world, metadata

    def _base_plan(self, base_goal_family: str, objects: List[Dict]):
        base_goal_family = base_goal_family.lower()
        states = []
        goal = ""
        success_check: Dict[str, any] = {}
        subgoals: List[Dict[str, Any]] = []

        if base_goal_family in ("putin", "put_in"):
            obj = objects[0]["name"]
            container = objects[1]["name"]
            states = [
                {"name": "grasp_object", "primitive": "grasp", "params": {"object": obj}},
                {"name": "place_into", "primitive": "place_in", "params": {"object": obj, "container": container}},
                {"name": "release_object", "primitive": "release", "params": {"object": obj}},
            ]
            goal = f"In({obj}, {container})"
            success_check = {
                "method": "combined",
                "operator": "and",
                "conditions": [
                    {"type": "in", "object": obj, "container": container},
                ],
            }
            subgoals = [
                {"type": "reach", "object": obj},
                {"type": "held", "object": obj},
                {"type": "in", "object": obj, "container": container},
            ]
        elif base_goal_family in ("grasp", "grasp_object"):
            obj = objects[0]["name"]
            states = [
                {"name": "grasp_object", "primitive": "grasp", "params": {"object": obj}},
            ]
            goal = f"Held({obj})"
            success_check = {"method": "simple", "conditions": [{"type": "held", "object": obj}]}
            subgoals = [
                {"type": "reach", "object": obj},
                {"type": "held", "object": obj},
            ]
        elif base_goal_family in ("puton", "put_on"):
            obj, support = objects[0]["name"], objects[1]["name"]
            states = [
                {"name": "grasp_object", "primitive": "grasp", "params": {"object": obj}},
                {"name": "lift_object", "primitive": "lift", "params": {"object": obj, "height": 0.07}},
                {"name": "place_on_support", "primitive": "place_on", "params": {"object": obj, "support": support}},
                {"name": "release_object", "primitive": "release", "params": {"object": obj}},
            ]
            goal = f"On({obj}, {support})"
            success_check = {
                "method": "combined",
                "operator": "and",
                "conditions": [{"type": "on", "object": obj, "support": support}],
            }
            subgoals = [
                {"type": "reach", "object": obj},
                {"type": "held", "object": obj},
                {"type": "on", "object": obj, "support": support},
            ]
        elif base_goal_family in ("insert", "insertion"):
            peg, hole = objects[0]["name"], objects[1]["name"]
            states = [
                {"name": "grasp_peg", "primitive": "grasp", "params": {"object": peg}},
                {
                    "name": "insert_peg",
                    "primitive": "insert",
                    "params": {"peg": peg, "hole": hole, "depth": 0.05, "angle_tol": 0.2},
                },
                {"name": "release_peg", "primitive": "release", "params": {"object": peg}},
            ]
            goal = f"Inserted({peg}, {hole}, 0.05, 0.2)"
            success_check = {
                "method": "combined",
                "operator": "and",
                "conditions": [
                    {"type": "inserted", "peg": peg, "hole": hole, "depth": 0.05, "angle_tol": 0.2}
                ],
            }
            subgoals = [
                {"type": "reach", "object": peg},
                {"type": "held", "object": peg},
                {"type": "inserted", "peg": peg, "hole": hole, "depth": 0.05, "angle_tol": 0.2},
            ]
        else:
            raise ValueError(f"Unsupported base goal family: {base_goal_family}")

        return states, goal, success_check, subgoals

    # Wrapper application ----------------------------------------------------
    def _apply_closed_receptacle(
        self,
        controller: PrimitiveController,
        predicates: PredicateEvaluator,
        objects: List[Dict],
    ) -> List[Dict]:
        receptacles = [obj for obj in objects if "openable" in obj.get("tags", [])]
        if not receptacles:
            return []
        container = receptacles[0]
        articulation = container.get("articulation", {})
        joint_name = articulation.get("joint", f"{container['name']}_joint")
        closed_value = articulation.get("closed")
        if closed_value is None:
            limits = articulation.get("limits", [0.0, 0.2])
            closed_value = limits[0] if limits else 0.0
        controller.close_joint(joint_name=joint_name, target=closed_value)
        predicates.context.closed_joints.add(joint_name)
        predicates.context.open_joints.discard(joint_name)
        open_target = articulation.get("open")
        if open_target is None:
            limits = articulation.get("limits", [0.0, 0.2])
            open_target = limits[-1] if limits else closed_value + 0.2
        open_state = {
            "name": "open_container",
            "primitive": "open_joint",
            "params": {"joint": joint_name, "target": open_target},
        }
        return [open_state]

    def _apply_out_of_reach(
        self,
        controller: PrimitiveController,
        predicates: PredicateEvaluator,
        objects: List[Dict],
    ) -> List[Dict]:
        containers = [obj for obj in objects if "container" in obj.get("tags", [])]
        if not containers:
            return []
        container = containers[0]
        target_object = objects[0]["name"]
        predicates.context.unreachable_objects.add(target_object)
        deps = predicates.context.pull_dependencies.setdefault(container["name"], set())
        deps.add(target_object)
        pull_state = {
            "name": "pull_container",
            "primitive": "pull",
            "params": {"object": container["name"], "direction": [-1.0, 0.0, 0.0], "distance": 0.2},
        }
        return [pull_state]

    def _apply_occluded_object(
        self,
        controller: PrimitiveController,
        predicates: PredicateEvaluator,
        objects: List[Dict],
    ) -> List[Dict]:
        occluders = [obj for obj in objects if "occluder" in obj.get("tags", [])]
        targets = [obj for obj in objects if "graspable" in obj.get("tags", [])]
        if not occluders or not targets:
            return []
        blocker = occluders[0]["name"]
        target = targets[0]["name"]
        predicates.context.blocked_pairs.add((blocker, target))
        push_state = {
            "name": "unblock_object",
            "primitive": "unblock",
            "params": {"blocker": blocker, "target": target, "distance": 0.2},
        }
        return [push_state]

    # ---------------------------------------------------------------- checks
    def _assert_necessity(
        self,
        controller: PrimitiveController,
        predicates: PredicateEvaluator,
        states: List[Dict],
        goal: str,
        wrappers: List[str],
        base_world: DictWorldState,
        metadata: Dict[str, ObjectMetadata],
        base_context: PredicateContext,
    ) -> None:
        if not wrappers:
            return

        def evaluate_plan(subset: List[Dict]) -> bool:
            local_world = base_world.clone()
            local_context = self._clone_context(base_context)
            local_predicates = PredicateEvaluator(local_world, metadata, local_context)
            local_controller = PrimitiveController(local_world, local_predicates)
            for state in subset:
                primitive = state["primitive"]
                params = state.get("params", {})
                getattr(local_controller, primitive)(**params)
            return self._goal_satisfied(local_predicates, goal)

        full_success = evaluate_plan(states)
        if not full_success:
            raise ValueError("Generated plan does not satisfy the goal.")

        for idx, state in enumerate(states):
            if state["primitive"] in {"open_joint", "pull", "unblock"}:
                reduced = states[:idx] + states[idx + 1 :]
                if evaluate_plan(reduced):
                    raise ValueError(f"Wrapper step {state['name']} not necessary.")

    def _goal_satisfied(self, predicates: PredicateEvaluator, goal: str) -> bool:
        goal = goal.strip()
        if goal.startswith("In("):
            args = goal[3:-1].split(",")
            return predicates.in_region(args[0].strip(), args[1].strip())
        if goal.startswith("On("):
            args = goal[3:-1].split(",")
            return predicates.on(args[0].strip(), args[1].strip())
        if goal.startswith("Held("):
            obj = goal[5:-1].strip()
            return predicates.held(obj)
        if goal.startswith("Inserted("):
            parts = [item.strip() for item in goal[9:-1].split(",")]
            depth = float(parts[2]) if len(parts) > 2 else 0.02
            angle = float(parts[3]) if len(parts) > 3 else 0.2
            return predicates.inserted(parts[0], parts[1], depth=depth, angle_tol=angle)
        return False

    def _build_spec_dict(
        self,
        task_name: str,
        base_goal_family: str,
        objects: List[Dict],
        states: List[Dict],
        goal: str,
        wrappers: List[str],
        success_check: Dict,
        subgoals: Sequence[Dict[str, Any]],
    ) -> Dict:
        return {
            "task_name": task_name,
            "base_goal_family": base_goal_family,
            "description": f"Auto-generated {base_goal_family} task",
            "objects": objects,
            "goal": goal,
            "obstructions": wrappers,
            "states": states,
            "success_check": success_check,
            "expected_min_steps": len(states),
            "subgoals": list(subgoals),
        }

    def _augment_subgoals(
        self,
        subgoals: Sequence[Dict[str, Any]],
        wrappers: List[str],
        objects: List[Dict],
    ) -> List[Dict[str, Any]]:
        augmented = list(subgoals)
        objects_by_name = {obj["name"]: obj for obj in objects}

        def insert_once(entry: Dict[str, Any], *, after_type: Optional[str] = None) -> None:
            if entry in augmented:
                return
            if after_type is not None:
                for index, existing in enumerate(augmented):
                    if existing.get("type") == after_type:
                        augmented.insert(index + 1, entry)
                        return
            augmented.insert(0, entry)

        if "closed_receptacle" in wrappers:
            container = next((o["name"] for o in objects if "openable" in o.get("tags", [])), None)
            if container:
                insert_once({"type": "open", "object": container})

        if "out_of_reach_with_pull" in wrappers and objects:
            target_object = objects[0]["name"]
            insert_once({"type": "reachable", "object": target_object})

        if "occluded_object" in wrappers:
            target = next((o["name"] for o in objects if "graspable" in o.get("tags", [])), None)
            if target:
                insert_once({"type": "clear", "support": target}, after_type="open")

        return augmented

    def _clone_context(self, context: PredicateContext) -> PredicateContext:
        cloned = PredicateContext()
        cloned.held_objects = set(context.held_objects)
        cloned.anchored_objects = set(context.anchored_objects)
        cloned.free_objects = set(context.free_objects)
        cloned.unreachable_objects = set(context.unreachable_objects)
        cloned.invisible_objects = set(context.invisible_objects)
        cloned.blocked_pairs = set(context.blocked_pairs)
        cloned.open_joints = set(context.open_joints)
        cloned.closed_joints = set(context.closed_joints)
        cloned.occlusion_memory = set(context.occlusion_memory)
        cloned.reach_radius = context.reach_radius
        cloned.object_radii = dict(context.object_radii)
        cloned.pull_dependencies = {k: set(v) for k, v in context.pull_dependencies.items()}
        if context.gripper_position is not None:
            cloned.gripper_position = np.array(context.gripper_position, dtype=float)
        cloned.gripper_site = context.gripper_site
        return cloned

    def _plan_executes(
        self,
        world: DictWorldState,
        metadata: Dict[str, ObjectMetadata],
        context: PredicateContext,
        states: List[Dict],
        goal: str,
    ) -> bool:
        local_world = world.clone()
        local_context = self._clone_context(context)
        predicates = PredicateEvaluator(local_world, metadata, local_context)
        controller = PrimitiveController(local_world, predicates)
        for state in states:
            getattr(controller, state["primitive"])(**state.get("params", {}))
        return self._goal_satisfied(predicates, goal)


__all__ = ["TaskGenerator", "GeneratorConfig", "GeneratedTask"]
