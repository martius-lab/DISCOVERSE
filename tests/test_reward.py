import numpy as np

from discoverse.universal_manipulation.predicates import DictWorldState, PredicateContext, PredicateEvaluator
from discoverse.universal_manipulation.primitives import PrimitiveController
from discoverse.universal_manipulation.reward import RewardShaper, ShapingConfig


def build_world():
    world = DictWorldState()
    world.add_body("block", [0.0, 0.0, 0.05])
    world.add_body("support", [0.0, 0.0, 0.0])
    metadata = {}
    predicates = PredicateEvaluator(world, metadata, PredicateContext())
    controller = PrimitiveController(world, predicates)
    return world, predicates, controller


def test_reward_increases_with_progress():
    world, predicates, controller = build_world()
    predicates.context.gripper_position = np.array([0.5, 0.0, 0.0])
    subgoals = [
        ShapingConfig(type="reach", target=[0.0, 0.0, 0.05], threshold=0.5),
        ShapingConfig(type="grasp", object="block"),
    ]
    shaper = RewardShaper(predicates, subgoals, completion_bonus=0.0, step_penalty=0.0)

    initial_reward = shaper.step(0.02)
    assert np.isclose(initial_reward, 0.0)

    predicates.context.gripper_position = np.array([0.02, 0.0, 0.05])
    reward = shaper.step(0.02)
    assert reward > 0.0

    controller.grasp(object="block")
    reward_after_grasp = shaper.step(0.02)
    assert reward_after_grasp > 0.0


def test_success_reward_requires_hold_time():
    world, predicates, controller = build_world()
    predicates.context.gripper_position = np.array([0.0, 0.0, 0.05])
    controller.grasp(object="block")
    subgoals = [ShapingConfig(type="grasp", object="block")]
    shaper = RewardShaper(predicates, subgoals, completion_bonus=1.0, step_penalty=0.0, success_hold_time=0.1)

    reward_first = shaper.step(0.05)
    assert reward_first > 0.0

    reward_second = shaper.step(0.05)
    assert np.isclose(reward_second, 1.0)

    reward_third = shaper.step(0.05)
    assert np.isclose(reward_third, 0.0)
