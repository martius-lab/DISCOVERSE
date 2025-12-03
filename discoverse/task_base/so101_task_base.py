import os
import json
import shutil
import mujoco
import mediapy
import numpy as np

from discoverse.robots_env.so101_base import SO101Base, SO101Cfg
from discoverse.utils import get_body_tmat

def recoder_so101(save_path, act_lst, obs_lst, cfg):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time" : [o['time'] for o in obs_lst],
            "obs"  : {
                "jq" : [o['jq'] for o in obs_lst],
                "jv" : [o['jv'] for o in obs_lst],
                "jf" : [o['jf'] for o in obs_lst],
                "ep" : [o['ep'] for o in obs_lst],
                "eq" : [o['eq'] for o in obs_lst],
            },
            "act"  : act_lst,
        }
        json.dump(obj, fp)

    for id in cfg.obs_rgb_cam_id:
        if 'img' in obs_lst[0] and obs_lst[0]['img'] is not None:
            # Check if image data exists for this camera ID
            if len(obs_lst[0]['img']) > id:
                mediapy.write_video(os.path.join(save_path, f"cam_{id}.mp4"), [o['img'][id] for o in obs_lst], fps=cfg.render_set["fps"])

class SO101TaskBase(SO101Base):
    target_control = np.zeros(6)
    action_done_dict = {
        "arm"           : False,
        "gripper"       : False,
        "delay"         : False,
    }
    delay_cnt = 0
    reset_sig = False

    def __init__(self, config: SO101Cfg):
        self.target_control = np.zeros(6)
        
        # View into target_control for arm (first 5 joints) and gripper (last joint)
        # SO-101 usually has 6 DOF where the last one is gripper
        self.tctr_arm = self.target_control[:5] 
        self.tctr_gripper = self.target_control[5:6]

        super().__init__(config)

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl.copy()
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def domain_randomization(self):
        """Skip domain randomization for base class"""
        pass

    def updateControl(self, action):
        # Action should be 6D: 5 arm joints + 1 gripper
        if len(action) != self.na:
             pass
        
        # Clip action to limits
        for i in range(self.na):
            self.mj_data.ctrl[i] = np.clip(action[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])

    def checkActionDone(self):
        # Arm convergence (indices 0-4)
        arm_done = np.allclose(self.target_control[:5], self.sensor_arm_qpos[:5], atol=3e-2) and \
                   np.abs(self.sensor_arm_qvel[:5]).sum() < 0.1
        
        # Gripper convergence (index 5)
        # Grippers can be stalled while applying force, so qvel check might need to be loose or ignored if stalled
        gripper_done = np.allclose(self.target_control[5], self.sensor_arm_qpos[5], atol=5e-2) or \
                       (np.abs(self.sensor_arm_qvel[5]) < 0.01) # Stalled/Finished

        self.delay_cnt -= 1
        delay_done = (self.delay_cnt<=0)

        self.action_done_dict = {
            "arm"           : arm_done,
            "gripper"       : gripper_done,
            "delay"         : delay_done,
        }
        
        # For simple tasks, we might just return True or a combination
        return arm_done and gripper_done and delay_done

    def printMessage(self):
        super().printMessage()
        print("target control : ")
        print("    tctr_arm         = {}".format(np.array2string(self.tctr_arm, separator=", ")))
        print("    tctr_gripper     = {}".format(np.array2string(self.tctr_gripper, separator=", ")))

        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

    def check_success(self):
        raise NotImplementedError
