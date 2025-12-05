from openai import OpenAI
import os
import re
import yaml as pyyaml
from discoverse import DISCOVERSE_ROOT_DIR

open_api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=open_api_key)

def get_file_content(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()

def get_path_content(path: str, file_type: str) -> dict[str, str]:
    import os
    contents = []
    if file_type == "dir":
        contents = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(file_type):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, path)
                    contents.append(relative_path)
    return ",".join(contents)


current_file_dir = os.path.dirname(os.path.abspath(__file__))
library_objects_path = os.path.join(DISCOVERSE_ROOT_DIR, "models/meshes/library_objects")
objects_name = get_path_content(library_objects_path, "dir")

example_output_1_yaml = get_file_content(current_file_dir+"/../discoverse/configs/tasks/so101_pick_place_cube.yaml")
example_output_1_xml = get_file_content(current_file_dir+"/../models/mjcf/task_environments/pick_place_cube.xml")

example_output_2_yaml = get_file_content(current_file_dir+"/../discoverse/configs/tasks/cover_cup_yaml.yaml")
example_output_2_xml = get_file_content(current_file_dir+"/../models/mjcf/task_environments/cover_cup.xml")

example_output_3_yaml = get_file_content(current_file_dir+"/../discoverse/configs/tasks/pick_place_scissors_in_container.yaml")
example_output_3_xml = get_file_content(current_file_dir+"/../models/mjcf/task_environments/pick_place_scissors_in_container.xml")


project_description = f"""
You are helping to generate necessary xml and yaml files for a Python project. The project has a factory class that creates tasks from YAML configuration files.
Each YAML file should follow the structure described below:
# YAML Task Configuration Keywords and Options
## TOP-LEVEL KEYWORDS
### task_name (required)
- Type: string
- Description: Name of the task, must match XML filename in task_environments/
- Example: "pick_place_cube", "cover_cup", "stack_block"
### robot_name (required)
- Type: string
- Options: "airbot_play", "so101", "mmk2", "tok2", "leaphand", "hand_with_arm"
- Description: Robot type to use
### description (optional)
- Type: string
- Description: Human-readable task description
### objects (optional)
- Type: list of strings
- Description: List of object body names that exist in the MuJoCo scene
- Example: ["block_green", "plate_white", "coffeecup_white"]
## robot_config (required)
### gs_models (optional)
- Type: dictionary mapping object names to .ply file paths
- Description: Gaussian Splatting models for rendering
- Format:
  ```yaml
  gs_models:
    background: "scene/lab3/point_cloud.ply"
    object_name: "object/model.ply"
  ```
### init_qpos (required)
- Type: list of floats
- Description: Initial joint positions, length must match robot's joint count
  - airbot_play: 7 values
  - so101: 6 values
  - mmk2: 12 values
  - tok2: 7 values
- Example: [0.0, -0.5, 0.8, 1.2, -0.8, 0.0]
### simulation (required)
- **timestep** (float): Simulation timestep (e.g., 0.001)
- **decimation** (int): Control decimation factor (e.g., 4)
- **sync** (bool): Synchronize simulation with real-time
- **headless** (bool): Run without visualization
### rendering (required)
- **fps** (int): Frames per second for recording (e.g., 30)
- **width** (int): Image width in pixels (e.g., 640)
- **height** (int): Image height in pixels (e.g., 480)
### camera_ids (required)
- Type: list of integers
- Description: Camera IDs for observation recording
- Example: [0] or [0, 1]
### save_mjb_and_task_config (optional)
- Type: bool
- Description: Save MuJoCo binary and task config
- Default: false
## randomization (optional)
### object_positions (optional)
- Type: list of dictionaries
- Format:
  ```yaml
  object_positions:
    - object: "object_name"
      position_range:
        x: 0.06  # +/- meters
        y: 0.06
        z: 0.01  # optional
  ```
### table_height (optional)
- Type: bool
- Description: Randomize table height
### table_config (required if table_height is true)
- **table_name** (string): Name of table body (usually "table")
- **affected_objects** (list): Objects that move with table
### table_texture (optional)
- Type: bool
- Description: Randomize table texture
### materials (optional)
- Type: list of strings
- Description: Material names to randomize
- Example: ["cube_texture", "plate_texture"]
### lighting (optional)
- Type: bool
- Description: Enable lighting randomization
### light_config (required if lighting is true)
- **direction** (bool): Randomize light direction
- **color** (bool): Randomize light color
- **active** (bool): Randomize which lights are active
## action_sequence (required)
List of motion primitives, each with:
- **type** (required): Primitive type
- **params** (required): Parameters for the primitive
### Motion Primitive Types:
#### 1. move_to_object
Move end-effector to an object
- **object** (string, required): Object body name
- **offset** (list [x,y,z], optional): World-frame offset in meters
- **rotation** (dict, optional):
  - **euler** (list [rx,ry,rz]): Euler angles
  - **seq** (string): Rotation sequence (e.g., "xyz")
  - **degrees** (bool): Use degrees instead of radians
- **gripper** (float, optional): Gripper position (0.0 = closed, 0.8 = open)
#### 2. move_to_position
Move to absolute or relative position
- **position** (list [x,y,z], required): Target position
- **rotation** (dict, optional): Same as move_to_object
- **relative** (bool, optional): Position relative to current pose
- **gripper** (float, optional): Gripper position
#### 3. track_object_offset
Track object with offset in object's local frame
- **object** (string, required): Object body name
- **local_offset** (list [x,y,z], optional): Offset in object's frame
- **rotation** (dict, optional): Rotation relative to object
- **gripper** (float, optional): Gripper position
#### 4. offset_current
Move relative to current end-effector pose
- **offset** (list [x,y,z], required): World-frame offset in meters
- **gripper** (float, optional): Gripper position
#### 5. grasp
Close gripper
- **position** (float, optional): Gripper close position (default: 0.0)
#### 6. release
Open gripper
- **position** (float, optional): Gripper open position (default: 0.0-1.0)
#### 7. delay
Wait for specified duration
- **duration** (float, required): Duration in seconds
## success_conditions (required)
### position_conditions (optional)
List of position-based success checks:
#### object_near_object
- **type**: "object_near_object"
- **object1** (string): First object name
- **object2** (string): Second object name
- **threshold** (float): Distance threshold in meters
#### planar_distance
- **type**: "planar_distance"
- **object1** (string): First object name
- **object2** (string): Second object name
- **axes** (string): Axes to check ("xy", "xz", "yz")
- **threshold** (float): Distance threshold in meters
### orientation_conditions (optional)
List of orientation-based success checks (format varies)
### custom_conditions (optional)
List of custom Python expressions:
- **expression** (string): Python expression that returns bool
- Can use: `get_body_tmat(mj_data, 'body_name')`, `mj_data`, numpy operations
- Example: `"get_body_tmat(mj_data, 'block')[2,3] > 0.75"`
## EXECUTION PARAMETERS
### max_time (optional)
- Type: float
- Description: Maximum task execution time in seconds
- Default: 15.0
### move_speed (optional)
- Type: float
- Description: Movement speed multiplier (0.0 to 1.0)
- Default: 0.65


Each YAML file also come with a .xml file that defines the environment for the task.



Your task is to generate two files based on the configure file .yaml and corresponded .xml files.
Requirements:
1. Return with the formate: 
<YAML_FILE>
...
<XML_FILE>
...
2. Task objects should be selected from the available objects listed below.
3. Ensure the YAML file follows the structure and options described above.
4. You will be given the general type of the task that you should generate. You need to select proper objects from the available objects list to complete the task.
5. The action_sequence in the YAML file should trying to finish the task goal logically step by step using motion primitives with reasonable parameters.
6. The initial object pose should be reachable by the lerobot so101 arm in the xml file.
7. The included file path in xml should change to `../object/library_objects/<OBJECT_NAME>.xml` rather than ../object/<OBJECT_NAME>.xml
8. materials in the yaml file should be named as <OBJECT_NAME>_material
9. Make sure each included mesh has its dependencies included in the xml file.
10. Objects should be placed high enough in the beginning to avoid penetration with the table.

Generate both file contents in one response, clearly separated.

Task related objects should be selected from the following available objects:
{objects_name} \n

Example Output 1:
<YAML_FILE>
{example_output_1_yaml}
<XML_FILE>
{example_output_1_xml}

Example Output 2:
<YAML_FILE>
{example_output_2_yaml}
<XML_FILE>
{example_output_2_xml}

Example Output 3:
<YAML_FILE>
{example_output_3_yaml}
<XML_FILE>
{example_output_3_xml}
"""
user_input = input("Enter your task description: ")


response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": project_description},
        {"role": "user", "content": user_input}
    ]
)

output = response.choices[0].message.content
yaml_match = re.search(r"<YAML_FILE>\s*(.*?)\s*<XML_FILE>", output, re.DOTALL)
xml_match = re.search(r"<XML_FILE>\s*(.*)", output, re.DOTALL)
XML_PATH = os.path.join(DISCOVERSE_ROOT_DIR, "models/mjcf/task_environments/")
YAML_PATH = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse/configs/tasks/")

if yaml_match and xml_match:
    yaml_content = yaml_match.group(1).strip()
    xml_content = xml_match.group(1).strip()

    # Extract task_name from YAML
    try:
        yaml_data = pyyaml.safe_load(yaml_content)
        task_name = yaml_data.get("task_name", "generated_task")
    except Exception:
        task_name = "generated_task"

    yaml_filename = YAML_PATH + f"{task_name}.yaml"
    xml_filename = XML_PATH + f"{task_name}.xml"

    with open(yaml_filename, "w") as f:
        f.write(yaml_content)
    with open(xml_filename, "w") as f:
        f.write(xml_content)
else:
    print("Could not parse YAML and XML sections from the response.")

print(f"Please run:\n python examples/run_yaml_task.py --config {yaml_filename} --data_set_size 2")

