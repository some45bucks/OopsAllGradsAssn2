# Monocular Odometry ROS

## Getting Started

This repository is configured to run both on a ROS robot directly and in a containerized VSCode environment.

### Running on the Robot

For running this package on the robot, ensure that `/opt/ros/<distro>/setup.bash` has been sourced
for the current shell session (either automatically or manually). Follow the below steps to run the ROS package:

1. Run `colcon build --symlink-install` from the root of the repository (colcon workspace root).
2. Run `source install/local_setup.bash`
3. TODO: Execute the appropriate script
 
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

#### Running

Start the ROS daemon by opening the `F1` palette and typing
`ROS: Start`. You can now use a regular terminal to work with ROS as if you were working on the robot platform (minus the robot
hardware. See previous section for details on how to run on a physical robot). Starting ROS typically only needs to be done once
after launching the dev container (check the VSCode status bar to see if there is a check next to the ROS version).

#### Using the ROS Extension in VSCode

VSCode's ROS extension includes a number of helpful features. If you don't want to worry about sourcing the local setup script for the workspace
when launching a terminal, you can launch a terminal using the `ROS: Create Terminal` command from the `F1` menu, or you can bind a keyboard
shortcut to launch a ROS terminal for you. (Note that when adding a new package to the workspace, it may not be automatically sourced, so you will either need
to manually source the local setup script or reload VSCode so that the extension rediscovers packages.)

If you want to run something using `ros2 run` or `ros2 launch` without needing to setup the environment for each terminal before running the commands manually, you can use the relevant `ROS:` commands in the `F1` menu to save some time. This is very convenient for development. (Adding
new packages to the workspace may require reloading VSCode to be detected
by this feature).
