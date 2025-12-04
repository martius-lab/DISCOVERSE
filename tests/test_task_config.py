from discoverse.universal_manipulation.task_config import TaskConfigLoader


def test_goal_compilation_and_operator():
    config = {
        "task_name": "goal_and",
        "description": "test",
        "states": [{"name": "dummy", "primitive": "noop"}],
        "goal": "In(block, container) ∧ Upright(block, 0.95)",
    }
    loader = TaskConfigLoader.from_dict(config)
    success = loader.success_check
    assert success["operator"] == "and"
    assert success["conditions"][0]["type"] == "in"
    assert success["conditions"][1]["type"] == "upright"


def test_goal_compilation_or():
    config = {
        "task_name": "goal_or",
        "description": "test",
        "states": [{"name": "dummy", "primitive": "noop"}],
        "goal": "Held(block) or On(block, support)",
    }
    loader = TaskConfigLoader.from_dict(config)
    assert loader.success_check["operator"] == "or"
    assert len(loader.success_check["conditions"]) == 2


def test_backwards_compatibility_without_extended_fields():
    original_config = {
        "task_name": "legacy",
        "description": "legacy task",
        "states": [{"name": "step1", "primitive": "move"}],
    }
    loader = TaskConfigLoader.from_dict(original_config)
    assert loader.objects_metadata == {}
