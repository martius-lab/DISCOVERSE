"""
Reward shaping utilities for DISCOVERSE universal manipulation tasks.

The spike implementation keeps the interface intentionally lightweight so it can
operate purely on the predicate layer without depending on a full MuJoCo
simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .predicates import PredicateEvaluator


@dataclass
class ShapingConfig:
    """Configuration for a single subgoal potential."""

    type: str
    object: Optional[str] = None
    support: Optional[str] = None
    container: Optional[str] = None
    target: Optional[List[float]] = None
    threshold: float = 0.05
    height: float = 0.05
    joint: Optional[str] = None
    goal_value: float = 0.2
    depth: float = 0.02
    angle_tol: float = 0.1


@dataclass
class RewardState:
    potential: float = 0.0
    success_hold_timer: float = 0.0
    completed: List[bool] = field(default_factory=list)
    success_bonus_granted: bool = False


class RewardShaper:
    """Potential-based reward shaper following the spec.md description."""

    def __init__(
        self,
        predicates: PredicateEvaluator,
        subgoals: List[ShapingConfig],
        *,
        completion_bonus: float = 1.0,
        step_penalty: float = 0.01,
        success_hold_time: float = 0.5,
    ):
        self.predicates = predicates
        self.subgoals = subgoals
        self.completion_bonus = completion_bonus
        self.step_penalty = step_penalty
        self.success_hold_time = success_hold_time
        self.state = RewardState(completed=[False] * len(subgoals))

    def reset(self) -> None:
        self.state = RewardState(completed=[False] * len(self.subgoals))

    def potential(self) -> float:
        return self.state.potential

    def step(self, dt: float = 0.02) -> float:
        """Compute shaped reward for the current timestep."""
        new_potential = self._compute_total_potential()
        reward = new_potential - self.state.potential - self.step_penalty
        self.state.potential = new_potential

        if all(self.state.completed):
            self.state.success_hold_timer += dt
            if (
                self.state.success_hold_timer >= self.success_hold_time
                and not self.state.success_bonus_granted
            ):
                reward += self.completion_bonus
                self.state.success_bonus_granted = True
        else:
            self.state.success_hold_timer = 0.0
            self.state.success_bonus_granted = False

        return float(reward)

    # ------------------------------------------------------------------ internals
    def _compute_total_potential(self) -> float:
        potentials = []
        for idx, subgoal in enumerate(self.subgoals):
            value, completed = self._subgoal_potential(subgoal)
            self.state.completed[idx] = completed
            potentials.append(value)
        if not potentials:
            return 0.0
        return float(np.mean(potentials))

    def _subgoal_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        goal_type = goal.type.lower()
        if goal_type == "reach":
            return self._reach_potential(goal)
        if goal_type == "grasp":
            return self._grasp_potential(goal)
        if goal_type == "lift":
            return self._lift_potential(goal)
        if goal_type in ("place_on", "placeon"):
            return self._place_on_potential(goal)
        if goal_type in ("place_in", "placein"):
            return self._place_in_potential(goal)
        if goal_type == "open":
            return self._open_potential(goal)
        if goal_type == "insert":
            return self._insert_potential(goal)
        return 0.0, False

    # ---------------------------------------------------------------- potentials
    def _reach_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        if goal.target is None:
            return 0.0, False
        gripper_pos = self.predicates.context.gripper_position
        if gripper_pos is None:
            gripper = np.zeros(3)
        else:
            gripper = np.asarray(gripper_pos, dtype=float)
        target = np.asarray(goal.target)
        distance = np.linalg.norm(gripper - target)
        normalized = 1.0 - min(distance / max(goal.threshold, 1e-6), 1.0)
        completed = distance <= goal.threshold
        return float(normalized), completed

    def _grasp_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        obj = goal.object
        held = self.predicates.held(obj)
        return (1.0 if held else 0.0), held

    def _lift_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        obj = goal.object
        if obj not in self.predicates.context.held_objects:
            return 0.0, False
        pos = self.predicates.state.get_body_position(obj)
        base_height = goal.threshold or 0.0
        progress = max(pos[2] - base_height, 0.0) / goal.height
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress, progress >= 1.0

    def _place_on_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        placed = self.predicates.on(goal.object, goal.support)
        if placed:
            return 1.0, True
        obj_pos = self.predicates.state.get_body_position(goal.object)
        sup_pos = self.predicates.state.get_body_position(goal.support)
        distance = np.linalg.norm(obj_pos[:2] - sup_pos[:2])
        normalized = 1.0 - min(distance / max(goal.threshold, 1e-6), 1.0)
        return float(normalized), False

    def _place_in_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        placed = self.predicates.in_region(goal.object, goal.container)
        if placed:
            return 1.0, True
        obj_pos = self.predicates.state.get_body_position(goal.object)
        cont_pos = self.predicates.state.get_body_position(goal.container)
        distance = np.linalg.norm(obj_pos[:2] - cont_pos[:2])
        normalized = 1.0 - min(distance / max(goal.threshold, 1e-6), 1.0)
        return float(normalized), False

    def _open_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        joint = goal.joint or goal.object
        value = self.predicates.state.get_joint_value(joint)
        progress = min(value / max(goal.goal_value, 1e-6), 1.0)
        completed = progress >= 1.0 or self.predicates.open(joint)
        return float(progress), completed

    def _insert_potential(self, goal: ShapingConfig) -> tuple[float, bool]:
        inserted = self.predicates.inserted(
            goal.object,
            goal.container,
            depth=goal.depth,
            angle_tol=goal.angle_tol,
        )
        if inserted:
            return 1.0, True
        peg_pos = self.predicates.state.get_body_position(goal.object)
        hole_pos = self.predicates.state.get_body_position(goal.container)
        depth_progress = max(hole_pos[2] - peg_pos[2], 0.0) / max(goal.depth, 1e-6)
        return float(np.clip(depth_progress, 0.0, 1.0)), False


__all__ = ["RewardShaper", "ShapingConfig"]
