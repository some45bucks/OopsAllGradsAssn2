# Monocular Odometry ROS

## Getting Started

This repository is configured to run both on a ROS robot directly and in a containerized VSCode environment.

You can also just run the monocular odometry code separately from everything else.

#### DATA
You can get the data here: https://drive.google.com/drive/folders/10pkFE-y-wk_2PBvocVuUoggc64GXfAjW?usp=share_link. just put the data folder in the root of the project

it has a video of the phone vide which shows a 90 degree turn and then the images and logs related to each trajectory file.

There are 4 trajectory files.

1. `trajectoryPhone10.png` this is the path from the 90 degree turn video at 10 fps
2. `trajectoryPhone20.png` this is the path from the 90 degree turn video at 20 fps
3. `trajectoryPyCam10Plus.png` this is the path from the robot driving in a square, where the red path is based on what the robot thinks is's velocity is. The green is constructed from the camera. This version is setting the cutoff to min 10 features.
4. `trajectoryPyCam40Plus.png` This version of the same path but with the cutoff at 40 features.

### Running the Monocular Odometry Code

To just run the monocular odometry code just run `pip install -r src/find_path_from_cam/requirements.txt` first to install the required packages

It is set to show the robot camera and path using using a feature cutoff of 40 (disregards the frame unless the feature count is higher than 40 and just assumes a straight direction). If you want to see the phone path just turn the varible `ROBOT = False` on line 200. to run it use `python3 src/find_path_from_camera/monocular_odometry.py`

### Running on the Robot

For running this package on the robot, ensure that `/opt/ros/<distro>/setup.bash` has been sourced
for the current shell session (either automatically or manually). Follow the below steps to run the ROS package:

1. Run `colcon build --symlink-install` from the root of the repository (colcon workspace root).
2. Run `source install/local_setup.bash`
3. Run `ros2 launch monocular_odometry teleop_with_camera_launch.xml`
4. If you have not already, install the teleop client python dependencies using `pip install -r teleop-controller-pc/requirements.txt` on the machine that will control the robot. It is recommended to create a virtual environment to install these dependencies into.
5. To run the teleop client on the machine and connect to the robot, run `python3 teleop-controller-pc/teleop-client.py 144.39.167.74`, substituting the example IP address for the IP address or hostname of the robot on the network.

--> The video and logs will start as soon as the robot moves for the first time <--
 
### Running in Development

This repository is configured to support VSCode's DevContainers extension. Install the VSCode Dev Containers extension
and Docker if you haven't already. After installing the extension and opening up the repository, you will either see
a prompt to reopen this repository in a development container, or you can press `F1` and type `Dev Containers: Reopen in Container`.
This will allow you to work inside a ROS2 environment that can run in docker on any platform. This container also
comes preloaded with the ROS2 VSCode extension.

#### Building

Next, build the repository by running `colcon build --symlink-install`. After the build completes, you will want to reload the
VSCode window so that the ROS extension picks up the built packages. This can be done by opening the command palette using `F1`,
followed by selecting `Developer: Reload Window`. You only have to do the "Reload Window" step during the first build or sometimes
when adding additional packages to the workspace. You may also need to make ROS update the python (or C++) imports for the python and C++ extension
if VSCode intellisense is not resolving import paths correctly, though this is usually done automatically when VSCode first loads or is reloaded.
(See the `ROS:` commands in `F1` menu for which commands will update these options in `.vscode/settings` if you need to manually trigger
these reloads)

#### Running ROS Commands

The integrated VSCode terminal will now connect to the docker container that is running the ROS environment. You can use the VSCode terminal
in the same way you would when running ROS on a bare metal install.

#### Using the ROS Extension in VSCode

VSCode's ROS extension includes a number of helpful features. If you don't want to worry about sourcing the local setup script for the workspace
when launching a terminal, you can launch a terminal using the `ROS: Create Terminal` command from the `F1` menu, or you can bind a keyboard
shortcut to launch a ROS terminal for you. (Note that when adding a new package to the workspace, it may not be automatically sourced, so you will either need
to manually source the local setup script or reload VSCode so that the extension rediscovers packages.)

If you want to run something using `ros2 run` or `ros2 launch` without needing to setup the environment for each terminal before running the commands manually, you can use the relevant `ROS:` commands in the `F1` menu to save some time. This is very convenient for development. (Adding
new packages to the workspace may require reloading VSCode to be detected
by this feature).
