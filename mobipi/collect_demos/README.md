# RBY1 Microwave Door Demo Collection

## Microwave fix

In `microwave.py`, replace the old direct `qpos` indexing:

```python
hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_microjoint")]
```

with:

```python
joint_name = f"{self.name}_microjoint"
hinge_qpos = float(sim.data.get_joint_qpos(joint_name))
```

This is needed because a MuJoCo joint ID is not guaranteed to be the same as its index in the `qpos` vector. Using `get_joint_qpos()` ensures RoboCasa reads the actual microwave hinge position correctly when checking the door state and task success.

## Run one microwave-closing demo

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python auto_close_microwave_door.py \
  --environment CloseSingleDoor \
  --layout 1 \
  --style 10 \
  --robot-x 0.8 \
  --robot-y -1.45 \
  --push-x-direction 1 \
  --push-lead 0.03 \
  --keep-open \
  --contact-z-offset 0 \
  --steps-per-waypoint 4 \
  --handle-clearance 0.053 \
  --arm left
```

## Collect multiple successful demonstrations

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python collect_close_door_demos.py \
  --count 10 \
  --layout 1 \
  --arm left \
  --robot-x 0.8 \
  --robot-y -1.45 \
  --push-x-direction 1 \
  --push-lead 0.03 \
  --steps-per-waypoint 4 \
  --handle-clearance 0.053 \
  --finish-angle 0.12
```