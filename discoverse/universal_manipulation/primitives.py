"""
Simplified manipulation primitives used during the spike phase.

These primitives operate on the lightweight DictWorldState abstraction and
maintain predicate context flags (e.g. held objects, open joints).  They are not
meant to be physically accurate – they provide deterministic geometric effects
that satisfy the behavioural contracts specified in spec.md, making them ideal
for unit tests and for the task generator prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .predicates import DictWorldState, PredicateContext, PredicateEvaluator


@dataclass
class PrimitiveResult:
    success: bool
    info: Optional[str] = None


class PrimitiveController:
    """Executes symbolic primitives by mutating world state directly."""

    def __init__(self, world: DictWorldState, predicates: PredicateEvaluator):
        self.world = world
        self.predicates = predicates

    # ------------------------------------------------------------------ utils
    def _gripper_pos(self) -> np.ndarray:
        pos = self.predicates.context.gripper_position
        if pos is not None:
            return np.asarray(pos, dtype=float)
        return np.zeros(3)

    def _set_gripper_pos(self, position: Sequence[float]) -> None:
        pos = np.asarray(position, dtype=float)
        self.predicates.context.gripper_position = pos

    def _object_pos(self, name: str) -> np.ndarray:
        return self.world.get_body_position(name)

    def _set_object_pos(self, name: str, pos: Sequence[float]) -> None:
        self.world.bodies[name].position = np.asarray(pos, dtype=float)

    def _object_quat(self, name: str) -> np.ndarray:
        return self.world.get_body_quat(name)

    def _set_joint(self, joint_name: str, value: float) -> None:
        self.world.set_joint(joint_name, value)

    # ---------------------------------------------------------------- reach
    def reach(self, site_name: str, tolerance: float = 0.01) -> PrimitiveResult:
        target = self.world.get_site_position(site_name)
        self._set_gripper_pos(target)
        if np.linalg.norm(self._gripper_pos() - target) <= tolerance:
            return PrimitiveResult(True)
        return PrimitiveResult(False, "gripper_not_near_target")

    # ---------------------------------------------------------------- grasp
    def grasp(self, object_name: Optional[str] = None, object: Optional[str] = None, proximity: float = 0.05) -> PrimitiveResult:
        target = object_name or object
        if target is None:
            return PrimitiveResult(False, "object_not_specified")
        if not self.predicates.reachable(target):
            return PrimitiveResult(False, "object_unreachable")
        for blocker, blocked_target in self.predicates.context.blocked_pairs:
            if blocked_target == target:
                return PrimitiveResult(False, "object_blocked")
        obj_pos = self._object_pos(target)
        gripper = self._gripper_pos()
        if np.linalg.norm(obj_pos - gripper) > proximity:
            return PrimitiveResult(False, "gripper_far_from_object")
        self.predicates.context.held_objects.add(target)
        return PrimitiveResult(True)

    def release(self, object_name: Optional[str] = None, object: Optional[str] = None) -> PrimitiveResult:
        target = object_name or object
        if target:
            self.predicates.context.held_objects.discard(target)
        return PrimitiveResult(True)

    # ---------------------------------------------------------------- lift
    def lift(
        self,
        object_name: Optional[str] = None,
        object: Optional[str] = None,
        height: float = 0.05,
    ) -> PrimitiveResult:
        target = object_name or object
        if target not in self.predicates.context.held_objects:
            return PrimitiveResult(False, "object_not_held")
        current = self._object_pos(target)
        lifted = current.copy()
        lifted[2] += abs(height)
        self._set_object_pos(target, lifted)
        return PrimitiveResult(True)

    # -------------------------------------------------------------- placements
    def place_on(
        self,
        object_name: Optional[str] = None,
        object: Optional[str] = None,
        support: Optional[str] = None,
        support_name: Optional[str] = None,
    ) -> PrimitiveResult:
        target = object_name or object
        support_target = support_name or support
        if target not in self.predicates.context.held_objects:
            return PrimitiveResult(False, "object_not_held")
        support_pos = self._object_pos(support_target)
        obj_pos = support_pos.copy()
        obj_pos[2] += 0.02
        self._set_object_pos(target, obj_pos)
        self.predicates.context.held_objects.discard(target)
        return PrimitiveResult(True)

    def place_in(
        self,
        object_name: Optional[str] = None,
        object: Optional[str] = None,
        container: Optional[str] = None,
        container_name: Optional[str] = None,
        *,
        height_offset: float = 0.02,
    ) -> PrimitiveResult:
        target = object_name or object
        container_target = container_name or container
        if target not in self.predicates.context.held_objects:
            return PrimitiveResult(False, "object_not_held")
        container_pos = self._object_pos(container_target)
        new_pos = container_pos.copy()
        new_pos[2] += height_offset
        if not self.predicates.open(container_target):
            return PrimitiveResult(False, "container_closed")
        self._set_object_pos(target, new_pos)
        self.predicates.context.held_objects.discard(target)
        return PrimitiveResult(True)

    # --------------------------------------------------------------- push/pull
    def push(
        self,
        object_name: Optional[str] = None,
        object: Optional[str] = None,
        direction: Sequence[float] = (1.0, 0.0, 0.0),
        distance: float = 0.1,
    ) -> PrimitiveResult:
        target = object_name or object
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        shift = direction * distance
        current = self._object_pos(target).copy()
        new_pos = current + shift
        self._set_object_pos(target, new_pos)
        return PrimitiveResult(True)

    def pull(
        self,
        object_name: Optional[str] = None,
        object: Optional[str] = None,
        direction: Sequence[float] = (-1.0, 0.0, 0.0),
        distance: float = 0.1,
    ) -> PrimitiveResult:
        target = object_name or object
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        shift = direction * distance
        current = self._object_pos(target).copy()
        self._set_object_pos(target, current + shift)

        ctx = self.predicates.context
        ctx.unreachable_objects.discard(target)
        for dependent in ctx.pull_dependencies.get(target, set()):
            ctx.unreachable_objects.discard(dependent)
        return PrimitiveResult(True)

    # -------------------------------------------------------------- joints
    def open_joint(self, joint_name: Optional[str] = None, joint: Optional[str] = None, target: float = 0.2) -> PrimitiveResult:
        name = joint_name or joint
        if name is None:
            return PrimitiveResult(False, "joint_not_specified")
        self._set_joint(name, target)
        self.predicates.context.open_joints.add(name)
        self.predicates.context.closed_joints.discard(name)
        return PrimitiveResult(True)

    def close_joint(self, joint_name: Optional[str] = None, joint: Optional[str] = None, target: float = 0.0) -> PrimitiveResult:
        name = joint_name or joint
        if name is None:
            return PrimitiveResult(False, "joint_not_specified")
        self._set_joint(name, target)
        self.predicates.context.closed_joints.add(name)
        self.predicates.context.open_joints.discard(name)
        return PrimitiveResult(True)

    # -------------------------------------------------------------- insertion
    def insert(
        self,
        peg_name: Optional[str] = None,
        peg: Optional[str] = None,
        hole_name: Optional[str] = None,
        hole: Optional[str] = None,
        *,
        depth: float,
        angle_tol: float,
    ) -> PrimitiveResult:
        peg_target = peg_name or peg
        hole_target = hole_name or hole
        if peg_target not in self.predicates.context.held_objects:
            return PrimitiveResult(False, "peg_not_held")
        if self.predicates.inserted(peg_target, hole_target, depth=depth, angle_tol=angle_tol):
            return PrimitiveResult(True)

        peg_pos = self._object_pos(peg_target)
        hole_pos = self._object_pos(hole_target)
        peg_pos[:2] = hole_pos[:2]
        peg_pos[2] = hole_pos[2] - depth
        self._set_object_pos(peg_target, peg_pos)

        if self.predicates.inserted(peg_target, hole_target, depth=depth, angle_tol=angle_tol):
            return PrimitiveResult(True)
        return PrimitiveResult(False, "alignment_failed")

    # -------------------------------------------------------------- unblock
    def unblock(
        self,
        blocker: Optional[str] = None,
        target: Optional[str] = None,
        distance: float = 0.2,
        shift: Optional[float] = None,
    ) -> PrimitiveResult:
        if blocker is None or target is None:
            return PrimitiveResult(False, "blocker_or_target_missing")
        amount = distance if shift is None else shift
        pos = self._object_pos(blocker).copy()
        pos[0] += amount
        self._set_object_pos(blocker, pos)
        self.predicates.context.blocked_pairs.discard((blocker, target))
        return PrimitiveResult(True)


__all__ = ["PrimitiveController", "PrimitiveResult"]
