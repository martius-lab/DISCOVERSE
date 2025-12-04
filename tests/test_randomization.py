import numpy as np
import pytest

import mujoco

from discoverse.universal_manipulation.randomization import SceneRandomizer


TEST_MJCF = """
<mujoco model="randomizer">
  <option timestep="0.01"/>
  <asset>
    <texture name="test_tex" type="2d" builtin="flat" rgb1="0.2 0.2 0.2" width="8" height="8"/>
  </asset>
  <worldbody>
    <body name="armbase" pos="0 0 0">
      <site name="armbase" pos="0 0 0"/>
    </body>
    <body name="table" pos="0 0 0">
      <geom type="box" size="0.3 0.3 0.02"/>
    </body>
    <body name="obj_a" pos="0.1 0 0.05">
      <joint type="free"/>
      <geom type="sphere" size="0.015"/>
    </body>
    <body name="obj_b" pos="-0.1 0 0.05">
      <joint type="free"/>
      <geom type="sphere" size="0.015"/>
    </body>
    <body name="drawer" pos="0.3 0 0.1">
      <joint name="drawer_joint" type="slide" axis="1 0 0" range="0 0.25"/>
      <geom type="box" size="0.05 0.05 0.02"/>
    </body>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"/>
    <camera name="main_cam" pos="0.5 0.0 0.5" quat="1 0 0 0"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def randomizer_env():
    model = mujoco.MjModel.from_xml_string(TEST_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    randomizer = SceneRandomizer(model, data)
    return randomizer, model, data


def test_object_randomization_respects_min_distance(randomizer_env):
    randomizer, model, data = randomizer_env
    config = {
        "objects": [
            {"name": "obj_a", "x_range": [0.0, 0.1], "y_range": [0.0, 0.1], "min_distance": 0.05},
            {"name": "obj_b", "x_range": [0.2, 0.3], "y_range": [0.0, 0.1], "min_distance": 0.05},
        ]
    }
    randomizer.exec_randomization(config, max_attempts=10)
    pos_a = data.body("obj_a").xpos
    pos_b = data.body("obj_b").xpos
    distance = np.linalg.norm(pos_a[:2] - pos_b[:2])
    assert distance >= 0.05


def test_object_randomization_fails_when_bounds_impossible(randomizer_env):
    randomizer, model, data = randomizer_env
    initial_pos = data.body("obj_a").xpos.copy()
    config = {
        "objects": [
            {"name": "obj_a", "x_range": [0.0, 0.0], "y_range": [0.0, 0.0], "min_distance": 0.1},
            {"name": "obj_b", "x_range": [0.0, 0.0], "y_range": [0.0, 0.0], "min_distance": 0.1},
        ]
    }
    success = randomizer._randomize_objects(config["objects"], max_attempts=3)
    assert not success
    assert np.allclose(data.body("obj_a").xpos, initial_pos)


def test_camera_randomization_bounds(randomizer_env):
    randomizer, model, data = randomizer_env
    config = {
        "cameras": {
            "main_cam": {
                "position_offset": [0.05, 0.05, 0.05],
                "orientation_offset": [0.05, 0.05, 0.05],
            }
        }
    }
    initial_pose = randomizer.initial_camera_poses["main_cam"]
    randomizer.exec_randomization(config)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
    camera = model.camera(cam_id)
    pos_delta = np.abs(camera.pos - initial_pose["pos"])
    assert np.all(pos_delta <= 0.05 + 1e-6)


def test_lighting_randomization(randomizer_env):
    randomizer, model, data = randomizer_env
    config = {
        "lighting": {
            "random_color": True,
            "random_active": True,
            "intensity_range": [0.1, 0.9],
        }
    }
    randomizer.exec_randomization(config)
    assert np.all(model.light_ambient >= 0.0) and np.all(model.light_ambient <= 1.0)
    assert np.all(model.light_diffuse >= 0.0) and np.all(model.light_diffuse <= 1.0)
    assert np.all(model.light_specular >= 0.0) and np.all(model.light_specular <= 1.0)
    assert np.sum(model.light_active) >= 1


def test_table_height_randomization(randomizer_env):
    randomizer, model, data = randomizer_env
    config = {
        "table_height": {
            "table_name": "table",
            "height_range": [0.0, 0.1],
            "affected_objects": ["obj_a"],
        }
    }
    randomizer.exec_randomization(config)
    table_z = model.body("table").pos[2]
    assert -0.1 <= table_z <= 0.0


def test_texture_randomization_updates_texture(randomizer_env, monkeypatch):
    randomizer, model, data = randomizer_env
    updates = {"viewer": False, "renderer": False}

    monkeypatch.setattr(SceneRandomizer, "_update_texture_viewer", lambda self, name: updates.__setitem__("viewer", True))
    monkeypatch.setattr(SceneRandomizer, "_update_texture_renderer", lambda self, name, img: updates.__setitem__("renderer", True))

    randomizer.viewer = object()
    randomizer.renderer = object()

    config = {
        "textures": {
            "objects": [
                {"name": "test_tex"},
            ]
        }
    }
    randomizer.exec_randomization(config)
    texture = model.texture("test_tex").data
    assert texture is not None
    assert updates["viewer"] and updates["renderer"]


def test_texture_randomization_without_viewer(randomizer_env, monkeypatch):
    randomizer, model, data = randomizer_env
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("_update_texture_* should not be called when viewer/renderer unset")

    monkeypatch.setattr(SceneRandomizer, "_update_texture_viewer", _should_not_be_called)
    monkeypatch.setattr(SceneRandomizer, "_update_texture_renderer", _should_not_be_called)
    config = {
        "textures": {
            "activate": True,
            "objects": [{"name": "test_tex"}],
        }
    }
    randomizer.viewer = None
    randomizer.renderer = None
    randomizer.exec_randomization(config)


def test_articulation_randomization_sets_joint(randomizer_env):
    randomizer, model, data = randomizer_env
    config = {
        "articulations": [
            {"joint": "drawer_joint", "qpos_range": [0.0, 0.02]},
        ]
    }
    randomizer.exec_randomization(config)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
    qaddr = model.jnt_qposadr[joint_id]
    value = data.qpos[qaddr]
    assert 0.0 <= value <= 0.02 + 1e-6


def test_randomization_seed_control(randomizer_env):
    _, model, _ = randomizer_env

    def run_with_seed(seed):
        np.random.seed(seed)
        model_local = mujoco.MjModel.from_xml_string(TEST_MJCF)
        data_local = mujoco.MjData(model_local)
        mujoco.mj_forward(model_local, data_local)
        randomizer_local = SceneRandomizer(model_local, data_local)
        config = {
            "objects": [
                {"name": "obj_a", "x_range": [0.0, 0.2], "y_range": [0.0, 0.2], "min_distance": 0.05},
                {"name": "obj_b", "x_range": [0.2, 0.4], "y_range": [0.0, 0.2], "min_distance": 0.05},
            ],
            "cameras": {
                "main_cam": {
                    "position_offset": [0.05, 0.05, 0.05],
                }
            },
        }
        randomizer_local.exec_randomization(config)
        return (
            data_local.body("obj_a").xpos.copy(),
            data_local.body("obj_b").xpos.copy(),
            model_local.camera(mujoco.mj_name2id(model_local, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")).pos.copy(),
        )

    pos_a1, pos_b1, cam1 = run_with_seed(123)
    pos_a2, pos_b2, cam2 = run_with_seed(123)
    assert np.allclose(pos_a1, pos_a2)
    assert np.allclose(pos_b1, pos_b2)
    assert np.allclose(cam1, cam2)

    pos_a3, pos_b3, cam3 = run_with_seed(None)
    differs = (
        not np.allclose(pos_a1, pos_a3)
        or not np.allclose(pos_b1, pos_b3)
        or not np.allclose(cam1, cam3)
    )
    assert differs
