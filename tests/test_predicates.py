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


def test_blocks_updates_with_unblock(primitive_controller):
    controller, predicates = primitive_controller
    controller.grasp(object="blocker")
    controller.place_on(object="blocker", support="block")
    assert predicates.blocks("blocker", "block")

    controller.unblock(blocker="blocker", target="block", distance=0.2)
    assert not predicates.blocks("blocker", "block")


def test_visible_respects_occluder(primitive_controller):
    controller, predicates = primitive_controller
    controller.world.bodies["block"].position = np.array([0.2, 0.0, 0.04])
    controller.world.bodies["blocker"].position = np.array([0.1, 0.0, 0.04])
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    assert predicates.visible("block")

    predicates.context.blocked_pairs.add(("blocker", "block"))
    assert not predicates.visible("block")

    controller.unblock(blocker="blocker", target="block", distance=0.2)
    predicates.context.blocked_pairs.clear()
    controller.world.bodies["blocker"].position = np.array([0.1, 0.0, 0.04])
    assert not predicates.visible("block")

    controller.world.bodies["blocker"].position = np.array([0.4, 0.0, 0.04])
    assert predicates.visible("block")


def test_at_region_matches_inside(primitive_controller):
    controller, predicates = primitive_controller
    controller.grasp(object="block")
    controller.open_joint(joint_name="container_joint", target=0.25)
    controller.place_in(object="block", container="container", height_offset=0.02)
    assert predicates.at("block", "container.inside", tolerance=0.05)


def test_reachable_toggles_with_context(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    assert predicates.reachable("block")
    predicates.context.unreachable_objects.add("block")
    assert not predicates.reachable("block")
    predicates.context.unreachable_objects.clear()
    assert predicates.reachable("block")
