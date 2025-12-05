# How to genereate new environment?
## Step1: Download library_objects
Download object set from `/al/backup/lerobot_hackathon/discoverse_meshes` and `/al/backup/lerobot_hackathon/discoverse_mjcfs`.

## Step2: Add soft link to code base

```
ln -s /path/to/discoverse_meshes/library_objects /path/to/DISCOVERSE/models/meshes/library_objects
ln -s /path/to/discoverse_mjcfs/library_objects /path/to/DISCOVERSE/models/mjcf/object/library_objects
```

## Step3: How to generate environment?
```
python3 scripts/text_to_tasks.py
```

It will print the command how to run the generated environment.

This scripts currently generate 2 files to the correct place with the same file name but different extend name:
`.yaml`: This is the task defination file.
`.xml`: This is the task environment file.

# WARNING:
**Remember to clean up the `models/mjcf/tmp` folder. If you rerun a task config, it will look into the tmp folder first to see if the mujoco xml exist. So if you rerun a changed config file, you need to clean up the tmp to force it regenerate a mujoco xml file for the final task.**