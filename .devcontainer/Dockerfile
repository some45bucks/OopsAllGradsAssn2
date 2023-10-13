FROM ros:iron
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID ros \
    && useradd --uid $USER_UID --gid $USER_GID -m ros -d /home/ros \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ros ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ros \
    && chmod 0440 /etc/sudoers.d/ros
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
# Auto-source ROS base environment and colcon_cd command
RUN echo "source /opt/ros/iron/setup.bash" >> /home/ros/.bashrc \
    && echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> /home/ros/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/ros/.bashrc
ENV SHELL /bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

USER ros
CMD ["/bin/bash"]