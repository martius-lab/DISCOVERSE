from pathlib import Path
import copy

import numpy as np
import pytest

from discoverse.universal_manipulation.generator import GeneratorConfig, TaskGenerator
from discoverse.universal_manipulation.predicates import PredicateContext, PredicateEvaluator
from discoverse.universal_manipulation.primitives import PrimitiveController
from discoverse.universal_manipulation.task_config import TaskConfigLoader


OBJECT_DEFINITION = [
    {
        "name": "block",
        "type": "block",
        "tags": ["graspable", "liftable"],
        "initial_position": [0.0, 0.0, 0.04],
        "geom_size": [0.02, 0.02, 0.02],
    },
    {
        "name": "container",
        "type": "container",
        "tags": ["container", "openable"],
        "initial_position": [0.2, 0.0, 0.0],
        "geom_size": [0.05, 0.05, 0.05],
        "articulation": {
            "joint": "container_joint",
            "limits": [0.0, 0.25],
            "handle_site": "handle_site",
            "closed": 0.0,
            "open": 0.2,
        },
        "regions": {
            "inside": {"body": "container", "type": "cylinder", "radius": 0.08, "height": 0.1}
        },
    },
    {
        "name": "support",
        "type": "platform",
        "tags": ["support_surface"],
        "initial_position": [0.3, 0.0, 0.0],
        "geom_size": [0.05, 0.05, 0.02],
    },
    {
        "name": "blocker",
        "type": "block",
        "tags": ["occluder", "pushable"],
        "initial_position": [0.0, 0.0, 0.08],
        "geom_size": [0.02, 0.02, 0.02],
    },
    {
        "name": "peg",
        "type": "peg",
        "tags": ["graspable", "insertable"],
        "initial_position": [0.0, 0.2, 0.1],
        "geom_size": [0.01, 0.01, 0.05],
    },
    {
        "name": "hole",
        "type": "hole",
        "tags": ["insert_target"],
        "initial_position": [0.0, 0.2, 0.0],
        "geom_size": [0.02, 0.02, 0.02],
    },
]


def _execute_plan(generator, objects, wrappers, states, goal):
    world, metadata = generator._initialise_world(objects)
    context = PredicateContext()
    if objects:
        context.gripper_position = np.array(objects[0].get("initial_position", [0.0, 0.0, 0.04]))
    predicates = PredicateEvaluator(world, metadata, context)
    controller = PrimitiveController(world, predicates)

    for wrapper in wrappers:
        method = {
            "closed_receptacle": generator._apply_closed_receptacle,
            "out_of_reach_with_pull": generator._apply_out_of_reach,
            "occluded_object": generator._apply_occluded_object,
        }.get(wrapper)
        if method:
            method(controller, predicates, objects)

    for state in states:
        primitive = state["primitive"]
        params = state.get("params", {})
        getattr(controller, primitive)(**params)

    return generator._goal_satisfied(predicates, goal)


def test_generate_put_in_task(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="put_in_basic",
        base_goal_family="put_in",
        objects=OBJECT_DEFINITION[:2],
        wrappers=[],
    )
    result = generator.generate(config)
    spec = result.specification
    assert spec["states"][0]["primitive"] == "grasp"
    assert spec["goal"].startswith("In")

    loader = TaskConfigLoader.from_dict(spec)
    assert loader.task_name == "put_in_basic"
    assert loader.success_check["operator"] == "and"

    assert _execute_plan(generator, config.objects, config.wrappers, spec["states"], spec["goal"])
    assert [sg["type"] for sg in spec["subgoals"]] == ["reach", "held", "in"]


def test_closed_receptacle_wrapper_requires_open(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="put_in_closed",
        base_goal_family="put_in",
        objects=OBJECT_DEFINITION[:2],
        wrappers=["closed_receptacle"],
    )
    result = generator.generate(config)
    spec = result.specification
    assert spec["states"][0]["primitive"] == "open_joint"

    assert spec["subgoals"][0]["type"] == "open"

    success_without_open = _execute_plan(generator, config.objects, config.wrappers, spec["states"][1:], spec["goal"])
    assert not success_without_open


def test_out_of_reach_wrapper_requires_pull(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="put_in_pull",
        base_goal_family="put_in",
        objects=OBJECT_DEFINITION[:2],
        wrappers=["out_of_reach_with_pull"],
    )
    result = generator.generate(config)
    spec = result.specification
    assert spec["states"][0]["primitive"] == "pull"

    assert spec["subgoals"][0]["type"] == "reachable"

    success_without_pull = _execute_plan(generator, config.objects, config.wrappers, spec["states"][1:], spec["goal"])
    assert not success_without_pull


def test_occluded_object_wrapper(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="grasp_occluded",
        base_goal_family="grasp",
        objects=[OBJECT_DEFINITION[0], OBJECT_DEFINITION[3]],
        wrappers=["occluded_object"],
    )
    result = generator.generate(config)
    spec = result.specification
    assert spec["states"][0]["primitive"] == "unblock"
    assert spec["subgoals"][0]["type"] == "clear"


def test_generated_yaml_saved(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="put_in_saved",
        base_goal_family="put_in",
        objects=OBJECT_DEFINITION[:2],
        wrappers=[],
    )
    result = generator.generate(config)
    assert result.yaml_path.exists()
    loaded = TaskConfigLoader.from_dict(result.specification)
    assert loaded.task_name == "put_in_saved"


def test_subgoals_are_structured_dicts(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="subgoal_structure",
        base_goal_family="put_on",
        objects=OBJECT_DEFINITION[:2] + [OBJECT_DEFINITION[2]],
        wrappers=[],
    )
    spec = generator.generate(config).specification
    assert all(isinstance(entry, dict) and "type" in entry for entry in spec["subgoals"])


def test_redundant_wrapper_rejected(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    objects = copy.deepcopy(OBJECT_DEFINITION[:2])
    articulation = objects[1]["articulation"]
    articulation["closed"] = articulation["open"]
    config = GeneratorConfig(
        task_name="redundant_wrapper",
        base_goal_family="put_in",
        objects=objects,
        wrappers=["closed_receptacle"],
    )
    with pytest.raises(ValueError):
        generator.generate(config)


def test_combined_wrappers_extend_subgoals(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    objects = copy.deepcopy(OBJECT_DEFINITION[:3])
    config = GeneratorConfig(
        task_name="combo_wrappers",
        base_goal_family="put_in",
        objects=objects,
        wrappers=["closed_receptacle", "occluded_object"],
    )
    spec = generator.generate(config).specification
    subgoal_types = [entry["type"] for entry in spec["subgoals"]]
    assert subgoal_types[0] == "open"
    assert "clear" in subgoal_types


def test_subgoal_sequence_matches_plan(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config = GeneratorConfig(
        task_name="sequence_check",
        base_goal_family="insert",
        objects=OBJECT_DEFINITION[-2:],
        wrappers=[],
    )
    spec = generator.generate(config).specification
    subgoal_types = [sg["type"] for sg in spec["subgoals"]]
    assert subgoal_types == ["reach", "held", "inserted"]


def test_insertion_feasibility(tmp_path):
    generator = TaskGenerator(output_dir=tmp_path)
    config_valid = GeneratorConfig(
        task_name="insert_valid",
        base_goal_family="insert",
        objects=OBJECT_DEFINITION[-2:],
        wrappers=[],
    )
    result = generator.generate(config_valid)
    assert _execute_plan(generator, config_valid.objects, [], result.specification["states"], result.specification["goal"])

    invalid_objects = [
        dict(OBJECT_DEFINITION[-2], initial_quat=[0.707, 0.0, 0.0, 0.707]),
        OBJECT_DEFINITION[-1],
    ]
    config_invalid = GeneratorConfig(
        task_name="insert_invalid",
        base_goal_family="insert",
        objects=invalid_objects,
        wrappers=[],
    )
    with pytest.raises(ValueError):
        generator.generate(config_invalid)
