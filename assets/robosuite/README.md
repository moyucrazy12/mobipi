# Importing RBY1 model into Robocasa

## How to import the model
Robocasa is built on top of [Robosuite](https://robosuite.ai/), which by itself is a simulation framework powered by MuJoCo.
Robocasa only introduces assets for the environment and different tasks, not for the robots.
All robots are defined in Robosuite. 
To keep this consistent, you will need to import all of the files in the current folder (`mobipi/assets/robosuite`) into their respective places in the robosuite library.
Note that the folder distribution mimics the one in the library, which should make copying the files over much easier.
Once coppied over, you need to add some `__init__.py`s and the robot should be available to use in Robocasa.

The library is located inside the conda environment, that is `~/anaconda3/envs/mobipi/lib/python3.10/site-packages/robosuite` (note that the path might be slightly different for you, depends on where your conda installation resides).
Open this folder, and copy over the files from this repo inside their respective folders in the robosuite library.

## Automated installation

Run the installer with the Conda environment directory as its only argument:

```bash
./assets/robosuite/install_rby1.sh ~/anaconda3/envs/mobipi
```

The script locates `robosuite` inside the environment, copies the RB-Y1 assets, applies the registration and mobile-base changes below, and creates one-time `.pre-rby1` backups of patched files. It is safe to run more than once.

## File specification
- `robosuite/models/assets/robots/rby1a`
  - This folder contains the geometry of the robot, i.e. the `.xml` file and the meshes.
  - Both the meshes and the `.xml` file were obtained from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
  - The `.xml` file was later changed to contain 3 cameras (head and 2 eye-in-hand cameras), and some coordinate frames required by robosuite were added.
  - **NOTE**: at the time of writing, the cameras were just added at random, and are for sure not in the correct position.
- `robosuite/models/assets/bases/rby1_mount.xml`
  - mostly placeholder file containing empty geometry and 1 coordinate frame.
  - This is required because robosuite treats the robot manipulators and bases seperate, but since RBY1 already contains a base, we need to add just an empty one.
- `robosuite/models/robots/manipulators/rby1_robot.py`
  - A python wrapper class for the robot (inspired by the Tiago robot class).
- `robosuite/models/grippers/rby1_gripper.py`
  - A wrapper connecting the gripper actions with the model.
- `robosuite/controllers/config/robots/default_rby1.json`
  - A config containing controller configuration for the RBY1 robot.
  - Once again inspired by the Tiago config.
  - **NOTE**: a lot of the values (damping, PD coefficients, etc.) were just coppied over from the Tiago config, and may need changing in the future.

## Steps:
1.) copy over the files at the top

2.) Go to `robosuite/models/robots/manipulators/__init__.py` and add this line
```python
from .rby1_robot import RBY1
```
3.) Go to  `robosuite/models/grippers/__init__.py` and add an import 
```python
from .rby1_gripper import RBY1Gripper
```
and a mapping
```python
GRIPPER_MAPPING ={
    ...
    "RBY1Gripper": RBY1Gripper,
    ...
}
```
4.) Go to `robosuite/robots/__init__.py` and add RBY1 to the mapping
```python
ROBOT_CLASS_MAPPING = {
    ...
    "RBY1": WheeledRobot,
    ...
}
```

5.) Go to `robosuite/controllers/parts/mobile_base/joint_vel.py`, find this block 
```python
base_action = np.copy([action[i] for i in [1, 0, 2]])
# input raw base action is delta relative to current pose of base
# controller expects deltas relative to initial pose of base at start of episode
# transform deltas from current base pose coordinates to initial base pose coordinates
x, y = base_action[0:2]

# do the reverse of theta rotation
base_action[0] = x * np.cos(theta) + y * np.sin(theta)
base_action[1] = -x * np.sin(theta) + y * np.cos(theta)
```
and replace it with 
```python
if jnt_dim == 3:
    base_action = np.copy([action[i] for i in [1, 0, 2]])
    # input raw base action is delta relative to current pose of base
    # controller expects deltas relative to initial pose of base at start of episode
    # transform deltas from current base pose coordinates to initial base pose coordinates
    x, y = base_action[0:2]

    # do the reverse of theta rotation
    base_action[0] = x * np.cos(theta) + y * np.sin(theta)
    base_action[1] = -x * np.sin(theta) + y * np.cos(theta)
else:
    base_action = np.asarray(action).copy()
```

## Verification
To ensure the model was imported correctly, run the `rby1_import_test.py` file, which opens a Robosuite environment called 'Lift'.
You should see the robot from the front view executing a waving maneuver.
