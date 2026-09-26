# Gaussian splatting in Mobipi
## How does the Gaussian splatting work?
1. `mobipi/scene_model/collect_images.py`
	- This script loads a given Robocasa environment and samples a defined number of images
		- Note that the environment contains the robot as well
	- It samples valid base positions of the robot where (without obstacle collision - there is a parameter in the robot base class that specifies the circle radius around the robot where no obstacle can be when it spawns)
	- For each position, it randomly sets the heading (yaw angle)
	- In each position, it has a set of predefined heights and pitch angles.
	- It renders images of the environment using the simulation, and saves following information which is needed to construct a Nerfstudio dataset:
		- cam_pos - camera position in the world frame
		- cam_xmat - rotation matrix of the camera (w.r.t. world frame)
		- cam_fov - fov of the camera
		- The 3D sampled image
		- (robot_pos - the sampled position, not used by the pipeline)
	- It also constructs a pointcloud of the scene by
		- Constructing a pointcloud of each image/render utilizing the depth camera
		- sampling a number of points from each image pc and stacking them
		- subsampling to get a predefined number of points
	- At the end of the script, it saves the collected data as a Nerfstudio dataset, and begins training by calling 
  ```python
  os.system(f'ns-train splatfacto-big --data {data_dir} --max-num-iterations 30000 --output-dir {os.path.join(data_dir, "model")} --project_name "" --experiment_name "" --viewer.quit-on-train-completion True --pipeline.model.background_color "white" --viewer.websocket-port $((RANDOM % 1001 + 7000))')
  ```
1. NerfStudio training
	- [Nerfstudio docs](https://docs.nerf.studio/quickstart/first_nerf.html)
	- Trained using the `ns-train` flag with NerfStudio's `splatfacto-big` model (implementation of gaussian splatting)
	- Once the model is trained, it is saved in the `mobipi/scene_model/scene_data` folder (subfolder `model` in each layout)
	- The model can be then used to render images given camera views, which is used in the Mobipi scoring method
## How to collect our own dataset?
- **In simulation**
	- We can reuse the `collect_images.py` script, as we can get the camera2world transforms in the simulator (small changes may be needed based on how we create the environment, but the script is robot agnostic by design)
- **In real life**
	- We need to construct a nerfstudio compatible dataset, that means obtain a set of pictures and corresponding camera transforms in the world frame.
	- We can use nerfstudio's `ns-process-data` script, which processes either a video or folder of images into a compatible dataset. This script uses [COLMAP](https://colmap.github.io/) (Structure-from-Motion library) to get the transforms of the pictures.
	- !!!!**THERE IS A BUG IN THE TRAINING SCRIPT** if you have less then 500 pictures, the dataset is saved on the gpu, where `torch.compile` is used to turn camera poses into a view matrix. This does not work, and will destroy the poses. So either use a big enough dataset, or set  `TORCHDYNAMO_DISABLE=1` as env variable before running the training.
- Note that appart from the Gaussian splatting model, the Mobipi method needs a pointcloud for collision detection and navigation (this can also be extracted from the nerfstudio's model, although I think we can create a better one from the depth cameras).
