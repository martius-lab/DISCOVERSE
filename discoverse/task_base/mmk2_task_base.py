import os
import json
import shutil
import mujoco
import mediapy
import numpy as np

from discoverse.robots import MMK2FIK
from discoverse.utils import get_body_tmat
from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg
import pickle


def recoder_mmk2(save_path, act_lst, obs_lst, cfg, state_lst=None, overview_only=False):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
                "base_position": [o["base_position"] for o in obs_lst],
                "base_orientation_wxyz": [o["base_orientation"] for o in obs_lst],
            },
            "act": act_lst,
        }
        json.dump(obj, fp)

    # Save MuJoCo states if provided (for later playback)
    # if state_lst is not None:
    #     np.savez_compressed(os.path.join(save_path, "states.npz"), states=state_lst)
    if state_lst is not None:
        with open(os.path.join(save_path, "mujoco_states.pkl"), "wb") as f:
            pickle.dump(state_lst, f)
    # Save videos
    if overview_only:
        # Only save overview camera (typically camera 0)
        if len(cfg.obs_rgb_cam_id) > 0:
            mediapy.write_video(
                os.path.join(save_path, "overview.mp4"),
                [o["img"][cfg.obs_rgb_cam_id[0]] for o in obs_lst],
                fps=cfg.render_set["fps"],
            )
    else:
        # Save all cameras
        for id in cfg.obs_rgb_cam_id:
            mediapy.write_video(
                os.path.join(save_path, f"cam_{id}.mp4"),
                [o["img"][id] for o in obs_lst],
                fps=cfg.render_set["fps"],
            )


class MMK2TaskBase(MMK2Base):
    arm_action = "pick"
    target_control = np.zeros(19)
    joint_move_ratio = np.zeros(19)
    action_done_dict = {
        "slide"         : False,
        "head"          : False,
        "left_arm"      : False,
        "left_gripper"  : False,
        "right_arm"     : False,
        "right_gripper" : False,
        "delay"         : False,
    }
    delay_cnt = 0
    reset_sig = False
    cam_id = 0

    gripper_done_limit = 0.35
    set_left_arm_new_target = False
    set_right_arm_new_target = False

    def __init__(self, config: MMK2Cfg):
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)

        self.arm_action_init_position = np.array([
            [0.223,  0.21, 1.07055],
            [0.223, -0.21, 1.07055],
        ])

        self.lft_arm_target_pose = self.arm_action_init_position[0].copy()
        self.rgt_arm_target_pose = self.arm_action_init_position[1].copy()
        if hasattr(self.config, "io_dim") and self.config.io_dim == 17:
            self.io_dim = 17
        else:
            self.io_dim = 19

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl.copy()
        self.lft_arm_target_pose = self.arm_action_init_position[0].copy()
        self.rgt_arm_target_pose = self.arm_action_init_position[1].copy()
        self.set_left_arm_new_target = False
        self.set_right_arm_new_target = False
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def get_tmat_wrt_mmk2base(self, pose):
        tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
        if pose.shape == (4,4):
            return np.linalg.inv(tmat_mmk2) @ pose
        else:
            return (np.linalg.inv(tmat_mmk2) @ np.append(pose, 1))[:3]

    def setArmEndTarget(self, target_pose, arm_action, arm, q_ref, a_rot):
        rq = MMK2FIK().get_armjoint_pose_wrt_footprint(target_pose, arm_action, arm, self.tctr_slide[0], q_ref, a_rot)
        if arm == "l":
            self.tctr_left_arm[:] = rq
            self.set_left_arm_new_target = True
        elif arm == "r":
            self.tctr_right_arm[:] = rq
            self.set_right_arm_new_target = True

    def updateControl(self, action):
        # assert len(action) == self.io_dim
        if self.io_dim == 19:
            self.mj_data.ctrl[:self.njctrl] = np.clip(action[:self.njctrl], self.mj_model.actuator_ctrlrange[:self.njctrl,0], self.mj_model.actuator_ctrlrange[:self.njctrl,1])
        elif self.io_dim == 17: 
            self.mj_data.ctrl[2:self.njctrl] = np.clip(action[:self.io_dim], self.mj_model.actuator_ctrlrange[2:self.njctrl,0], self.mj_model.actuator_ctrlrange[2:self.njctrl,1])
        else:
            raise ValueError(f"Wrong action dim{self.io_dim}")

    def checkActionDone(self):
        slide_done = np.allclose(self.tctr_slide, self.sensor_slide_qpos, atol=3e-2) and np.abs(self.sensor_slide_qvel).sum() < 1e-2
        head_done = np.allclose(self.tctr_head, self.sensor_head_qpos, atol=3e-2) and np.abs(self.sensor_head_qvel).sum() < 0.1

        if self.set_left_arm_new_target:
            left_arm_done = np.allclose(self.lft_arm_target_pose, self.sensor_lftarm_ep, atol=3e-2) and np.abs(self.sensor_lft_arm_qvel).sum() < 0.1
            if left_arm_done:
                self.set_left_arm_new_target = False
        else:
            left_arm_done = True
        left_gripper_done = np.allclose(self.tctr_lft_gripper, self.sensor_lft_gripper_qpos, atol=self.gripper_done_limit) and np.abs(self.sensor_lft_gripper_qvel).sum() < 0.125

        if self.set_right_arm_new_target:
            right_arm_done = np.allclose(self.rgt_arm_target_pose, self.sensor_rgtarm_ep, atol=3e-2) and np.abs(self.sensor_rgt_arm_qvel).sum() < 0.1
            if right_arm_done:
                self.set_right_arm_new_target = False
        else:
            right_arm_done = True
        right_gripper_done = np.allclose(self.tctr_rgt_gripper, self.sensor_rgt_gripper_qpos, atol=self.gripper_done_limit) and np.abs(self.sensor_rgt_gripper_qvel).sum() < 0.125

        self.delay_cnt -= 1
        delay_done = (self.delay_cnt<=0)

        self.action_done_dict = {
            "slide"         : slide_done,
            "head"          : head_done,
            "left_arm"      : left_arm_done,
            "left_gripper"  : left_gripper_done,
            "right_arm"     : right_arm_done,
            "right_gripper" : right_gripper_done,
            "delay"         : delay_done,
        }
        return slide_done and head_done and left_arm_done and left_gripper_done and right_arm_done and right_gripper_done and delay_done

    def printMessage(self):
        super().printMessage()
        print("target control : ")
        print("    tctr_base        = {}".format(np.array2string(self.tctr_base, separator=", ")))
        print("    tctr_slide       = {}".format(np.array2string(self.tctr_slide, separator=", ")))
        print("    tctr_head        = {}".format(np.array2string(self.tctr_head, separator=", ")))
        print("    tctr_left_arm    = {}".format(np.array2string(self.tctr_left_arm, separator=", ")))
        print("    tctr_lft_gripper = {}".format(np.array2string(self.tctr_lft_gripper, separator=", ")))
        print("    tctr_right_arm   = {}".format(np.array2string(self.tctr_right_arm, separator=", ")))
        print("    tctr_rgt_gripper = {}".format(np.array2string(self.tctr_rgt_gripper, separator=", ")))
        print("    lft  endpoint    = {}".format(np.array2string(self.lft_arm_target_pose, separator=", ")))
        print("    rgt  endpoint    = {}".format(np.array2string(self.rgt_arm_target_pose, separator=", ")))

        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

    def check_success(self):
        raise NotImplementedError

    def domain_randomization(self):
        pass

    def visual_domain_randomization(self):

        # add a camera and render its get its seg mask
        if not hasattr(self, "overview_cam"):
            self.overview_cam = mujoco.MjvCamera()
            self.overview_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Position: (2, 0, 1.5), looking at (0, 0, 0)
            # Calculate lookat, distance, azimuth, elevation
            # Or just set the camera pose directly if we can?
            # MjvCamera uses lookat/distance/azimuth/elevation for FREE camera usually.
            # But we can try to set it up manually.

            # Let's use lookat and distance
            self.overview_cam.lookat[:] = [0, 0, 0]
            self.overview_cam.distance = 5 # sqrt(2^2 + 1.5^2) = 2.5
            self.overview_cam.elevation = -36.87 # -asin(1.5/2.5) * 180/pi
            self.overview_cam.azimuth = 80 # looking from +x to -x

        if self.overview_cam not in self.config.obs_seg_cam_id:
            self.config.obs_seg_cam_id.append(self.overview_cam)
        segmask = self.getSegMask(self.overview_cam)
        rgb = self.getRgbImg(self.overview_cam)

        obj_id, geom_id = segmask[:,:,0], segmask[:,:,1]

        # save image

        from discoverse.utils import get_random_texture
        print("randomize table texture")
        self.update_texture("tc_texture", get_random_texture())
        self.random_material("tc_texture")

        self.random_material("kiwi_texture")
        self.random_material("grid")

        # random light
        print("randomize light")
        self.random_light(write_color=False)
        return

    def random_material(self, mtl_name, random_color=False, emission=False):
        try:
            if random_color:
                self.mj_model.material(mtl_name).rgba[:3] = np.random.rand(3)
            if emission:
                self.mj_model.material(mtl_name).emission = np.random.rand()
            self.mj_model.material(mtl_name).specular = np.random.rand()
            self.mj_model.material(mtl_name).reflectance = np.random.rand()
            self.mj_model.material(mtl_name).shininess = np.random.rand()
        except KeyError:
            print(f"Warning: material {mtl_name} not found")

    def random_light(self, random_dir=True, random_color=True, random_active=True, write_color=False):
        if write_color:
            for i in range(self.mj_model.nlight):
                self.mj_model.light_ambient[i, :] = np.random.random()
                self.mj_model.light_diffuse[i, :] = np.random.random()
                self.mj_model.light_specular[i, :] = np.random.random()
        elif random_color:
            self.mj_model.light_ambient[...] = np.random.random(size=self.mj_model.light_ambient.shape)
            self.mj_model.light_diffuse[...] = np.random.random(size=self.mj_model.light_diffuse.shape)
            self.mj_model.light_specular[...] = np.random.random(size=self.mj_model.light_specular.shape)

        if random_active:
            self.mj_model.light_active[:] = np.int32(np.random.rand(self.mj_model.nlight) > 0.5).tolist()

        if np.sum(self.mj_model.light_active) == 0:
            self.mj_model.light_active[np.random.randint(self.mj_model.nlight)] = 1

        self.mj_model.light_pos[:,:2] = self.mj_model.light_pos0[:,:2] + np.random.normal(scale=0.3, size=self.mj_model.light_pos[:,:2].shape)
        self.mj_model.light_pos[:,2] = self.mj_model.light_pos0[:,2] + np.random.normal(scale=0.2, size=self.mj_model.light_pos[:,2].shape)

        if random_dir:
            self.mj_model.light_dir[:] = np.random.random(size=self.mj_model.light_dir.shape) - 0.5
            self.mj_model.light_dir[:,2] *= 2.0
            self.mj_model.light_dir[:] = self.mj_model.light_dir[:] / np.linalg.norm(self.mj_model.light_dir[:], axis=1, keepdims=True)
            self.mj_model.light_dir[:,2] = -np.abs(self.mj_model.light_dir[:,2])