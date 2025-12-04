import numpy as np

from discoverse.universal_manipulation.primitives import PrimitiveController


def test_reach_updates_gripper_position(primitive_controller):
    controller, predicates = primitive_controller
    result = controller.reach(site_name="handle_site", tolerance=0.01)
    assert result.success
    target = controller.world.get_site_position("handle_site")
    assert np.linalg.norm(predicates.context.gripper_position - target) <= 0.01


def test_grasp_requires_proximity(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.gripper_position = np.array([1.0, 1.0, 1.0])
    result_fail = controller.grasp(object="block")
    assert not result_fail.success

    predicates.context.gripper_position = np.array([0.0, 0.0, 0.04])
    result_success = controller.grasp(object="block")
    assert result_success.success
    assert predicates.held("block")


def test_lift_requires_object_held(primitive_controller):
    controller, predicates = primitive_controller
    controller.grasp(object="block")
    start_height = controller.world.get_body_position("block")[2]
    result = controller.lift(object="block", height=0.07)
    assert result.success
    new_height = controller.world.get_body_position("block")[2]
    assert new_height >= start_height + 0.07


def test_place_on_sets_on_and_releases(primitive_controller):
    controller, predicates = primitive_controller
    controller.grasp(object="block")
    controller.lift(object="block", height=0.05)
    controller.place_on(object="block", support="support")
    assert predicates.on("block", "support")
    assert not predicates.held("block")


def test_place_in_sets_in_and_releases(primitive_controller):
    controller, predicates = primitive_controller
    controller.grasp(object="block")
    controller.lift(object="block", height=0.05)
    controller.open_joint(joint_name="container_joint", target=0.25)
    controller.place_in(object="block", container="container", height_offset=0.02)
    assert predicates.in_region("block", "container")
    assert not predicates.held("block")


def test_push_and_pull_shift_objects(primitive_controller):
    controller, _ = primitive_controller
    start = controller.world.get_body_position("blocker").copy()
    controller.push(object="blocker", direction=[1.0, 0.0, 0.0], distance=0.1)
    after_push = controller.world.get_body_position("blocker")
    assert np.isclose(after_push[0] - start[0], 0.1, atol=1e-3)

    controller.pull(object="blocker", direction=[-1.0, 0.0, 0.0], distance=0.05)
    after_pull = controller.world.get_body_position("blocker")
    assert np.isclose(after_pull[0] - start[0], 0.05, atol=1e-3)


def test_open_and_close_joint_update_context(primitive_controller):
    controller, predicates = primitive_controller
    open_result = controller.open_joint(joint_name="container_joint", target=0.2)
    assert open_result.success
    assert controller.world.joints["container_joint"] == 0.2
    assert "container_joint" in predicates.context.open_joints

    close_result = controller.close_joint(joint_name="container_joint", target=0.0)
    assert close_result.success
    assert controller.world.joints["container_joint"] == 0.0
    assert "container_joint" in predicates.context.closed_joints


def test_insert_sets_inserted_predicate(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.gripper_position = np.array([0.0, 0.2, 0.1])
    controller.grasp(object="peg")
    controller.insert(peg="peg", hole="hole", depth=0.05, angle_tol=0.2)
    assert predicates.inserted("peg", "hole", depth=0.05, angle_tol=0.2)


def test_unblock_clears_blockers(primitive_controller):
    controller, predicates = primitive_controller
    predicates.context.blocked_pairs.add(("blocker", "block"))
    controller.unblock(blocker="blocker", target="block", distance=0.2)
    assert ("blocker", "block") not in predicates.context.blocked_pairs
