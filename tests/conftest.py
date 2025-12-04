import sys
import types

import numpy as np
import pytest

# Provide a lightweight stub for optional mink dependency used by runtime modules.
if "mink" not in sys.modules:
    mink_stub = types.SimpleNamespace(move_mocap_to_frame=lambda *args, **kwargs: None)
    sys.modules["mink"] = mink_stub
if "av" not in sys.modules:
    av_module = types.ModuleType("av")
    av_module.open = lambda *args, **kwargs: None
    av_video = types.ModuleType("av.video")
    av_module.video = av_video
    sys.modules["av"] = av_module
    sys.modules["av.video"] = av_video

from discoverse.universal_manipulation.predicates import (
    DictWorldState,
    ObjectMetadata,
    PredicateContext,
    PredicateEvaluator,
)
from discoverse.universal_manipulation.primitives import PrimitiveController


@pytest.fixture
def base_world():
    """Provide a reusable manipulation world with common objects."""
    world = DictWorldState()
    world.add_body("table", [0.0, 0.0, 0.0], geom_size=[0.2, 0.2, 0.02])
    world.add_body("block", [0.0, 0.0, 0.04])
    world.add_body("support", [0.2, 0.0, 0.0])
    world.add_body("container", [0.4, 0.0, 0.0])
    world.add_body("peg", [0.0, 0.2, 0.1])
    world.add_body("hole", [0.0, 0.2, 0.0])
    world.add_body("blocker", [0.0, 0.0, 0.08])
    world.add_site("handle_site", [0.5, 0.0, 0.0])

    metadata = {
        "table": ObjectMetadata(name="table", type="platform", tags={"support_surface"}),
        "block": ObjectMetadata(name="block", type="block", tags={"graspable", "liftable"}),
        "support": ObjectMetadata(name="support", type="platform", tags={"support_surface"}),
        "container": ObjectMetadata(
            name="container",
            type="container",
            tags={"container", "receptacle", "openable"},
            regions={
                "inside": {
                    "body": "container",
                    "type": "cylinder",
                    "radius": 0.08,
                    "height": 0.1,
                }
            },
            articulation_joint="container_joint",
            articulation_limits=(0.0, 0.25),
            handle_site="handle_site",
        ),
        "peg": ObjectMetadata(name="peg", type="peg", tags={"insertable"}),
        "hole": ObjectMetadata(name="hole", type="hole", tags={"insert_target"}),
        "blocker": ObjectMetadata(name="blocker", type="block", tags={"occluder", "pushable"}),
    }
    world.set_joint("container_joint", 0.0)

    predicates = PredicateEvaluator(world, metadata, PredicateContext())
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    return world, predicates


@pytest.fixture
def primitive_controller(base_world):
    world, predicates = base_world
    controller = PrimitiveController(world, predicates)
    return controller, predicates
