import numpy as np

from discoverse.universal_manipulation.predicates import PredicateEvaluator


def test_grasp_and_release_toggle_held(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.gripper_position = np.array([0.3, 0.3, 0.3])

    assert not predicates.held("block")
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    controller.grasp(object="block")
    assert predicates.held("block")

    controller.release(object="block")
    assert not predicates.held("block")


def test_clear_updates_with_blocker(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.08])

    controller.grasp(object="block")
    controller.place_on(object="block", support="table")
    assert not predicates.clear("table")

    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    controller.grasp(object="block")
    controller.place_on(object="block", support="support")
    assert predicates.clear("table")


def test_anchored_and_free_flags(base_world):
    world, predicates = base_world
    predicates.context.anchored_objects.add("table")
    predicates.context.free_objects.add("block")

    assert predicates.anchored("table")
    assert predicates.free("block")


def test_access_requires_all_components(primitive_controller):
    controller, predicates = primitive_controller
    obj = "block"

    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    controller.grasp(object="blocker")
    controller.place_on(object="blocker", support=obj)
    assert not predicates.access(obj)  # Not clear

    controller.unblock(blocker="blocker", target=obj, distance=0.1)
    controller.grasp(object=obj)
    controller.place_on(object=obj, support="support")
    predicates.context.invisible_objects.add(obj)
    assert not predicates.access(obj)

    predicates.context.invisible_objects.clear()
    predicates.context.unreachable_objects.add(obj)
    assert not predicates.access(obj)

    predicates.context.unreachable_objects.clear()
    assert predicates.access(obj)
