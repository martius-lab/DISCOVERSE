import mujoco
import numpy as np
from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class SO101Cfg(BaseConfig):
    mjcf_file_path = "mjcf/lerobot_so101/xml/so101_tabletop_manipulation_generated.xml" #This one is the full scene with table and objects
    decimation     = 4
    timestep       = 0.001
    sync           = True
    headless       = False
    init_key       = "0"
    render_set     = {
        "fps"    : 30,
        "width"  : 640,
        "height" : 480,
    }
    obs_rgb_cam_id  = [0]
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6"]
    
    obj_list       = []
    use_gaussian_renderer = False 
    
class SO101Base(SimulatorBase):
    def __init__(self, config: SO101Cfg):
        self.nj = 6 
        self.na = 6 
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
            self.init_joint_ctrl = np.zeros(self.na)
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.na)

        self.sensor_arm_qpos = self.mj_data.sensordata[:6]
        self.sensor_arm_qvel = self.mj_data.sensordata[6:12]
        self.sensor_arm_force = self.mj_data.sensordata[12:18]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[18:21]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[21:25]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[25:28]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[28:31]
        self.sensor_endpoint_acc = self.mj_data.sensordata[31:34]

    def printMessage(self):
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.na] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        for i in range(self.na):
            self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(self.mj_data.ctrl[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])
 
    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_arm_qpos.tolist(),
            "jv"   : self.sensor_arm_qvel.tolist(),
            "jf"   : self.sensor_arm_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "img"  : self.img_rgb_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = SO101Cfg()
    exec_node = SO101Base(cfg)

    obs = exec_node.reset()
    
    action = exec_node.init_joint_ctrl[:exec_node.na]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
