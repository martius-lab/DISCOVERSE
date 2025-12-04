"""
Predicate evaluation utilities for DISCOVERSE universal manipulation tasks.

This module provides a lightweight world-state abstraction that can be backed
either by a real MuJoCo model/data pair or by a synthetic in-memory world
representation (useful for unit tests and task generation logic).  Predicates
are implemented as composable functions over that abstraction so they can be
shared across runtime success checks, planners, generators, and reward shaping.

The spike implementation intentionally keeps the math simple – distance and
axis-alignment checks rely on coarse geometric heuristics that are sufficient
for robust unit testing.  More precise geometric queries (ray casting, contact
tests, etc.) can be swapped in during the stabilisation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:  # MuJoCo is optional for pure unit tests that rely on DictWorldState.
    import mujoco
except Exception:  # pragma: no cover - mujoco may be unavailable in CI.
    mujoco = None


# ---------------------------------------------------------------------------
# World state abstractions


class WorldStateAdapter(Protocol):
    """Protocol describing the minimal state queries predicates rely on."""

    def get_body_position(self, name: str) -> np.ndarray: ...

    def get_body_quat(self, name: str) -> np.ndarray: ...

    def get_site_position(self, name: str) -> np.ndarray: ...

    def get_site_quat(self, name: str) -> np.ndarray: ...

    def get_joint_value(self, joint_name: str) -> float: ...

    def get_geom_aabb(self, geom_name: str) -> Tuple[np.ndarray, np.ndarray]: ...


class MutableWorldState(WorldStateAdapter, Protocol):
    """Protocol extending WorldStateAdapter with mutation helpers."""

    def set_body_position(self, name: str, position: Sequence[float]) -> None: ...

    def translate_body(self, name: str, delta: Sequence[float]) -> None: ...

    def set_joint_value(self, joint_name: str, value: float) -> None: ...


class MuJoCoStateAdapter(MutableWorldState):
    """Adapter exposing MuJoCo model/data through the WorldStateAdapter API."""

    def __init__(self, mj_model, mj_data):
        if mujoco is None:
            raise RuntimeError("MuJoCo is required to use MuJoCoStateAdapter.")
        self._model = mj_model
        self._data = mj_data

    # --- body/site helpers -------------------------------------------------

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def get_body_position(self, name: str) -> np.ndarray:
        body = self._data.body(name)
        return np.asarray(body.xpos)

    def get_body_quat(self, name: str) -> np.ndarray:
        quat = self._data.body(name).xquat
        # MuJoCo uses wxyz order; Rotation expects xyzw.
        return np.asarray([quat[1], quat[2], quat[3], quat[0]])

    def get_site_position(self, name: str) -> np.ndarray:
        return np.asarray(self._data.site(name).xpos)

    def get_site_quat(self, name: str) -> np.ndarray:
        mat = self._data.site(name).xmat.reshape(3, 3)
        quat = Rotation.from_matrix(mat).as_quat()
        return np.asarray(quat)

    def get_joint_value(self, joint_name: str) -> float:
        joint_id = self._joint_id(joint_name)
        qpos_addr = self._model.jnt_qposadr[joint_id]
        return float(self._data.qpos[qpos_addr])

    def get_geom_aabb(self, geom_name: str) -> Tuple[np.ndarray, np.ndarray]:
        geom_id = self._geom_id(geom_name)
        aabb_min = self._data.geom_xpos[geom_id] - self._model.geom_size[geom_id]
        aabb_max = self._data.geom_xpos[geom_id] + self._model.geom_size[geom_id]
        return np.asarray(aabb_min), np.asarray(aabb_max)

    # -- Mutation helpers ---------------------------------------------------
    def set_body_position(self, name: str, position: Sequence[float]) -> None:
        pos = np.asarray(position, dtype=float)
        self._data.body(name).xpos[:] = pos

    def translate_body(self, name: str, delta: Sequence[float]) -> None:
        delta_vec = np.asarray(delta, dtype=float)
        body = self._data.body(name)
        body.xpos[:] = body.xpos + delta_vec

    def set_joint_value(self, joint_name: str, value: float) -> None:
        joint_id = self._joint_id(joint_name)
        qpos_addr = self._model.jnt_qposadr[joint_id]
        self._data.qpos[qpos_addr] = float(value)


@dataclass
class WorldObject:
    """Lightweight record representing a rigid body."""

    name: str
    position: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    geom_size: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.02]))


@dataclass
class WorldSite:
    """Record for a named site (pose expressed via quaternion)."""

    name: str
    position: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))


@dataclass
class DictWorldState(MutableWorldState):
    """
    Simple in-memory adapter used in tests and in the generator spike.

    Notes:
        - Body poses are expressed in world coordinates.
        - Quaternions follow xyzw order, matching scipy.spatial.
        - Geoms share the same bounds as bodies unless explicitly set.
    """

    bodies: Dict[str, WorldObject] = field(default_factory=dict)
    sites: Dict[str, WorldSite] = field(default_factory=dict)
    joints: Dict[str, float] = field(default_factory=dict)
    geom_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def add_body(
        self,
        name: str,
        position: Sequence[float],
        quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        geom_size: Sequence[float] = (0.02, 0.02, 0.02),
    ) -> None:
        self.bodies[name] = WorldObject(
            name=name,
            position=np.asarray(position, dtype=float),
            quat=np.asarray(quat, dtype=float),
            geom_size=np.asarray(geom_size, dtype=float),
        )

    def add_site(
        self,
        name: str,
        position: Sequence[float],
        quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.sites[name] = WorldSite(
            name=name,
            position=np.asarray(position, dtype=float),
            quat=np.asarray(quat, dtype=float),
        )

    def set_body_position(self, name: str, position: Sequence[float]) -> None:
        if name not in self.bodies:
            raise KeyError(f"Body '{name}' not defined in DictWorldState.")
        self.bodies[name].position = np.asarray(position, dtype=float)

    def translate_body(self, name: str, delta: Sequence[float]) -> None:
        if name not in self.bodies:
            raise KeyError(f"Body '{name}' not defined in DictWorldState.")
        self.bodies[name].position = self.bodies[name].position + np.asarray(delta, dtype=float)

    def set_joint_value(self, joint_name: str, value: float) -> None:
        self.joints[joint_name] = float(value)

    def set_joint(self, joint_name: str, value: float) -> None:
        self.set_joint_value(joint_name, value)

    def set_geom_bounds(
        self,
        geom_name: str,
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        self.geom_bounds[geom_name] = (
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def clone(self) -> "DictWorldState":
        new_world = DictWorldState()
        for body in self.bodies.values():
            new_world.add_body(body.name, body.position.copy(), body.quat.copy(), body.geom_size.copy())
        for site in self.sites.values():
            new_world.add_site(site.name, site.position.copy(), site.quat.copy())
        new_world.joints = self.joints.copy()
        new_world.geom_bounds = {
            name: (bounds[0].copy(), bounds[1].copy()) for name, bounds in self.geom_bounds.items()
        }
        return new_world

    # -- Adapter API -------------------------------------------------------
    def get_body_position(self, name: str) -> np.ndarray:
        if name not in self.bodies:
            raise KeyError(f"Body '{name}' not defined in DictWorldState.")
        return self.bodies[name].position

    def get_body_quat(self, name: str) -> np.ndarray:
        if name not in self.bodies:
            raise KeyError(f"Body '{name}' not defined in DictWorldState.")
        return self.bodies[name].quat

    def get_site_position(self, name: str) -> np.ndarray:
        if name not in self.sites:
            raise KeyError(f"Site '{name}' not defined in DictWorldState.")
        return self.sites[name].position

    def get_site_quat(self, name: str) -> np.ndarray:
        if name not in self.sites:
            raise KeyError(f"Site '{name}' not defined in DictWorldState.")
        return self.sites[name].quat

    def get_joint_value(self, joint_name: str) -> float:
        if joint_name not in self.joints:
            raise KeyError(f"Joint '{joint_name}' not defined in DictWorldState.")
        return self.joints[joint_name]

    def get_geom_aabb(self, geom_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if geom_name in self.geom_bounds:
            return self.geom_bounds[geom_name]
        if geom_name in self.bodies:
            body = self.bodies[geom_name]
            return body.position - body.geom_size, body.position + body.geom_size
        raise KeyError(f"Geometry '{geom_name}' not defined in DictWorldState.")


# ---------------------------------------------------------------------------
# Context / metadata


@dataclass
class ObjectMetadata:
    """Rich metadata for objects defined in YAML configs."""

    name: str
    type: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    regions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    articulation_joint: Optional[str] = None
    articulation_limits: Optional[Tuple[float, float]] = None
    articulation_open_value: Optional[float] = None
    articulation_closed_value: Optional[float] = None
    handle_site: Optional[str] = None


@dataclass
class PredicateContext:
    """Mutable runtime context shared across predicates and primitives."""

    held_objects: Set[str] = field(default_factory=set)
    anchored_objects: Set[str] = field(default_factory=set)
    free_objects: Set[str] = field(default_factory=set)
    unreachable_objects: Set[str] = field(default_factory=set)
    invisible_objects: Set[str] = field(default_factory=set)
    blocked_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    open_joints: Set[str] = field(default_factory=set)
    closed_joints: Set[str] = field(default_factory=set)
    occlusion_memory: Set[Tuple[str, str]] = field(default_factory=set)
    gripper_position: Optional[np.ndarray] = None
    gripper_site: Optional[str] = None
    reach_radius: float = 0.5
    object_radii: Dict[str, float] = field(default_factory=dict)
    pull_dependencies: Dict[str, Set[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predicate evaluator


class PredicateEvaluator:
    """Collection of predicate helpers operating on a state adapter."""

    def __init__(
        self,
        state: WorldStateAdapter,
        objects: Optional[Dict[str, ObjectMetadata]] = None,
        context: Optional[PredicateContext] = None,
    ):
        self._state = state
        self._objects = objects or {}
        self._context = context or PredicateContext()

    # -- Context management -------------------------------------------------
    @property
    def context(self) -> PredicateContext:
        return self._context

    @property
    def state(self) -> WorldStateAdapter:
        return self._state

    def set_gripper_position(self, position: Sequence[float]) -> None:
        self._context.gripper_position = np.asarray(position, dtype=float)

    def update_object_metadata(self, metadata: Iterable[ObjectMetadata]) -> None:
        self._objects = {obj.name: obj for obj in metadata}

    # -- Utility helpers ----------------------------------------------------
    def _get_object_radius(self, object_name: str, default: float = 0.05) -> float:
        if object_name in self._context.object_radii:
            return self._context.object_radii[object_name]

        meta = self._objects.get(object_name)
        if meta and "radius" in meta.regions.get("surface", {}):
            return float(meta.regions["surface"]["radius"])

        # Fallback to half of bbox diagonal derived from geom size (if available)
        try:
            lower, upper = self._state.get_geom_aabb(object_name)
            size = upper - lower
            radius = float(np.linalg.norm(size[:2]) / 2.0)
            self._context.object_radii[object_name] = radius
            return radius
        except Exception:
            pass

        if isinstance(self._state, DictWorldState) and object_name in self._state.bodies:
            size = self._state.bodies[object_name].geom_size
            radius = float(np.linalg.norm(size[:2]) / 2.0)
            self._context.object_radii[object_name] = radius
            return radius

        return default

    def _gripper_position(self) -> np.ndarray:
        ctx = self._context
        if ctx.gripper_position is not None:
            return ctx.gripper_position
        if ctx.gripper_site is not None:
            return self._state.get_site_position(ctx.gripper_site)
        # Default origin near workspace.
        return np.zeros(3)

    def _parse_region_identifier(self, default_owner: str, identifier: str) -> Tuple[str, str]:
        if "." in identifier:
            owner, region = identifier.split(".", 1)
            return owner, region
        return default_owner, identifier

    def _get_region_definition(self, owner: str, region_name: str) -> Dict[str, Any]:
        meta = self._objects.get(owner)
        if not meta:
            return {}
        return meta.regions.get(region_name, {})

    def _region_center(self, owner: str, region_name: str) -> Optional[np.ndarray]:
        region = self._get_region_definition(owner, region_name)
        if not region:
            return None
        center = self._state.get_body_position(region.get("body", owner))
        offset = np.asarray(region.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        return center + offset

    def _occludes_line(self, blocker_name: str, start: np.ndarray, target: np.ndarray) -> bool:
        blocker_pos = self._state.get_body_position(blocker_name)
        segment = target - start
        length = np.linalg.norm(segment)
        if length < 1e-6:
            return False
        projection = np.dot(blocker_pos - start, segment) / (length ** 2)
        if projection <= 0.0 or projection >= 1.0:
            return False
        closest = start + projection * segment
        lateral = np.linalg.norm(blocker_pos - closest)
        return lateral <= self._get_object_radius(blocker_name) and np.linalg.norm(blocker_pos - start) < length

    # -- Core predicates ----------------------------------------------------
    def on(
        self,
        object_name: str,
        support_name: str,
        *,
        lateral_tolerance: float = 0.05,
        height_tolerance: float = 0.05,
    ) -> bool:
        obj_pos = self._state.get_body_position(object_name)
        support_pos = self._state.get_body_position(support_name)
        lateral_distance = np.linalg.norm(obj_pos[:2] - support_pos[:2])
        vertical_gap = obj_pos[2] - support_pos[2]
        return (
            lateral_distance <= lateral_tolerance
            and 0.0 <= vertical_gap <= height_tolerance
        )

    def in_region(
        self,
        object_name: str,
        container_name: str,
        *,
        region_name: str = "inside",
    ) -> bool:
        owner, resolved_region = self._parse_region_identifier(container_name, region_name)
        region = self._get_region_definition(owner, resolved_region)
        obj_pos = self._state.get_body_position(object_name)
        if not region:
            # Spherical approximation using container radius if known.
            radius = self._get_object_radius(owner, default=0.08)
            center = self._state.get_body_position(owner)
            return np.linalg.norm(obj_pos[:2] - center[:2]) <= radius and obj_pos[2] <= center[2] + radius

        region_type = region.get("type", "cylinder")
        center = self._state.get_body_position(region.get("body", owner))
        offset = np.asarray(region.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        center = center + offset
        if region_type == "cylinder":
            radius = float(region.get("radius", 0.08))
            height = float(region.get("height", 0.05))
            radial = np.linalg.norm(obj_pos[:2] - center[:2])
            height_ok = center[2] <= obj_pos[2] <= center[2] + height
            return radial <= radius and height_ok
        if region_type == "box":
            half_extents = np.asarray(region.get("half_extents", [0.05, 0.05, 0.05]))
            lower = center - half_extents
            upper = center + half_extents
            return np.all(obj_pos >= lower) and np.all(obj_pos <= upper)
        return False

    def near(self, object_name: str, target: Sequence[float], threshold: float = 0.05) -> bool:
        obj_pos = self._state.get_body_position(object_name)
        target = np.asarray(target, dtype=float)
        return np.linalg.norm(obj_pos - target) <= threshold

    def at(
        self,
        object_name: str,
        region_identifier: str,
        tolerance: float = 0.05,
    ) -> bool:
        owner, region_name = self._parse_region_identifier(object_name, region_identifier)
        center = self._region_center(owner, region_name)
        if center is None:
            return self.near(object_name, self._state.get_body_position(region_identifier), threshold=tolerance)
        obj_pos = self._state.get_body_position(object_name)
        return np.linalg.norm(obj_pos - center) <= tolerance

    def aligned(
        self,
        body_a: str,
        body_b: str,
        *,
        axis: Sequence[float],
        tolerance: float,
    ) -> bool:
        axis = np.asarray(axis, dtype=float)
        direction = axis / (np.linalg.norm(axis) + 1e-8)
        rot_a = Rotation.from_quat(self._state.get_body_quat(body_a))
        rot_b = Rotation.from_quat(self._state.get_body_quat(body_b))
        vec_a = rot_a.apply(direction)
        vec_b = rot_b.apply(direction)
        cos_angle = np.clip(np.dot(vec_a, vec_b), -1.0, 1.0)
        return cos_angle >= np.cos(tolerance)

    def upright(self, object_name: str, threshold: float = 0.95) -> bool:
        rot = Rotation.from_quat(self._state.get_body_quat(object_name))
        z_axis = rot.apply(np.array([0.0, 0.0, 1.0]))
        return float(z_axis[2]) >= threshold

    def held(self, object_name: str) -> bool:
        return object_name in self._context.held_objects

    def open(self, object_name: str, *, threshold: float = 1e-3) -> bool:
        metadata = self._objects.get(object_name)
        if metadata and metadata.articulation_joint:
            joint_value = self._state.get_joint_value(metadata.articulation_joint)
            if metadata.articulation_open_value is not None:
                ref = metadata.articulation_open_value
                base = metadata.articulation_closed_value if metadata.articulation_closed_value is not None else ref
                tolerance = max(threshold, 0.05 * abs(ref - base))
                return joint_value >= ref - tolerance
            if metadata.articulation_limits:
                qmin, qmax = metadata.articulation_limits
                span = abs(qmax - qmin)
                tolerance = max(threshold, 0.05 * span)
                return joint_value >= qmax - tolerance
            return joint_value > threshold
        # Fallback to context bookkeeping.
        return object_name in self._context.open_joints

    def closed(self, object_name: str, *, threshold: float = 1e-3) -> bool:
        metadata = self._objects.get(object_name)
        if metadata and metadata.articulation_joint:
            joint_value = self._state.get_joint_value(metadata.articulation_joint)
            if metadata.articulation_closed_value is not None:
                ref = metadata.articulation_closed_value
                base = metadata.articulation_open_value if metadata.articulation_open_value is not None else ref
                tolerance = max(threshold, 0.05 * abs(base - ref))
                return joint_value <= ref + tolerance
            if metadata.articulation_limits:
                qmin, qmax = metadata.articulation_limits
                span = abs(qmax - qmin)
                tolerance = max(threshold, 0.05 * span)
                return joint_value <= qmin + tolerance
            return abs(joint_value) <= threshold
        return object_name in self._context.closed_joints

    def reachable(self, object_name: str, radius: Optional[float] = None) -> bool:
        if object_name in self._context.unreachable_objects:
            return False
        gripper = self._gripper_position()
        obj_pos = self._state.get_body_position(object_name)
        reach_radius = radius if radius is not None else self._context.reach_radius
        return np.linalg.norm(obj_pos - gripper) <= reach_radius

    def visible(self, object_name: str) -> bool:
        ctx = self._context
        if object_name in ctx.invisible_objects:
            return False

        for blocker, target in ctx.blocked_pairs:
            if target == object_name:
                ctx.occlusion_memory.add((blocker, target))
                return False

        start = self._gripper_position()
        target_pos = self._state.get_body_position(object_name)
        # Remember occluder pairs that have historically blocked visibility so we
        # can re-evaluate them even after the wrapper-induced context flags are cleared.
        occluding_pairs: Set[Tuple[str, str]] = set(
            pair for pair in ctx.occlusion_memory if pair[1] == object_name
        )
        for name, meta in self._objects.items():
            if name == object_name or not meta:
                continue
            if "occluder" not in meta.tags:
                continue
            pair = (name, object_name)
            if pair in ctx.blocked_pairs or pair in ctx.occlusion_memory:
                occluding_pairs.add(pair)

        for blocker, _ in list(occluding_pairs):
            if self._occludes_line(blocker, start, target_pos):
                ctx.occlusion_memory.add((blocker, object_name))
                return False
            ctx.occlusion_memory.discard((blocker, object_name))
        return True

    def clear(self, support_name: str) -> bool:
        for other in self._objects.keys():
            if other == support_name:
                continue
            if self.on(other, support_name):
                return False
        return True

    def free(self, object_name: str) -> bool:
        if object_name in self._context.anchored_objects:
            return False
        if object_name in self._context.free_objects:
            return True
        # Default: objects are considered free.
        return True

    def anchored(self, object_name: str) -> bool:
        if object_name in self._context.anchored_objects:
            return True
        return not self.free(object_name)

    def blocks(self, blocker: str, target: str) -> bool:
        if (blocker, target) in self._context.blocked_pairs:
            return True
        blocker_pos = self._state.get_body_position(blocker)
        target_pos = self._state.get_body_position(target)
        lateral = np.linalg.norm(blocker_pos[:2] - target_pos[:2])
        vertical_overlap = blocker_pos[2] >= target_pos[2]
        return lateral <= self._get_object_radius(target) and vertical_overlap

    def inserted(
        self,
        peg: str,
        hole: str,
        *,
        depth: float,
        angle_tol: float,
    ) -> bool:
        peg_pos = self._state.get_body_position(peg)
        hole_pos = self._state.get_body_position(hole)
        depth_ok = hole_pos[2] - peg_pos[2] >= depth
        rot = Rotation.from_quat(self._state.get_body_quat(peg))
        peg_axis = rot.apply(np.array([0.0, 0.0, 1.0]))
        align = abs(np.dot(peg_axis, np.array([0.0, 0.0, 1.0])))
        return depth_ok and align >= np.cos(angle_tol)

    def access(self, object_name: str) -> bool:
        return self.reachable(object_name) and self.visible(object_name) and self.clear(object_name)


# Convenience functions -----------------------------------------------------

def build_metadata_map(objects_config: Sequence[dict]) -> Dict[str, ObjectMetadata]:
    """Convert raw YAML object entries into ObjectMetadata instances."""
    metadata = {}
    for obj in objects_config or []:
        tags = set(obj.get("tags", []))
        articulation = obj.get("articulation", {})
        regions = obj.get("regions", {})
        metadata[obj.get("name")] = ObjectMetadata(
            name=obj.get("name"),
            type=obj.get("type"),
            tags=tags,
            regions=regions,
            articulation_joint=articulation.get("joint"),
            articulation_limits=tuple(articulation.get("limits", [])) or None,
            handle_site=articulation.get("handle_site"),
        )
    return metadata


__all__ = [
    "PredicateEvaluator",
    "PredicateContext",
    "ObjectMetadata",
    "DictWorldState",
    "WorldStateAdapter",
    "MuJoCoStateAdapter",
    "build_metadata_map",
]
