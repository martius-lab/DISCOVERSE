import numpy as np

from discoverse.universal_manipulation.predicates import DictWorldState, ObjectMetadata, PredicateContext, PredicateEvaluator
from discoverse.universal_manipulation.primitives import PrimitiveController


def build_container_world():
    world = DictWorldState()
    world.add_body("container", [0.0, 0.0, 0.0])
    world.add_body("object", [0.1, 0.0, 0.05])
    metadata = {
        "container": ObjectMetadata(
            name="container",
            type="container",
            tags={"container", "openable"},
            articulation_joint="container_joint",
            articulation_limits=(0.0, 0.2),
            regions={
                "inside": {"body": "container", "type": "cylinder", "radius": 0.05, "height": 0.1}
            },
        )
    }
    world.set_joint("container_joint", 0.0)
    predicates = PredicateEvaluator(world, metadata, PredicateContext())
    controller = PrimitiveController(world, predicates)
    predicates.context.gripper_position = np.array([0.1, 0.0, 0.05])
    return world, predicates, controller


def test_place_in_requires_open_container():
    world, predicates, controller = build_container_world()
    controller.grasp(object="object")
    controller.place_in(object="object", container="container", height_offset=0.02)
    assert not predicates.in_region("object", "container")

    controller.open_joint(joint_name="container_joint", target=0.2)
    controller.place_in(object="object", container="container", height_offset=0.02)
    assert predicates.in_region("object", "container")


def test_in_region_boundary():
    world, predicates, _ = build_container_world()
    world.bodies["object"].position = np.array([0.05, 0.0, 0.05])
    assert predicates.in_region("object", "container")

    world.bodies["object"].position = np.array([0.051, 0.0, 0.05])
    assert not predicates.in_region("object", "container")
