#!/usr/bin/env bash

# Clean old ROS overlays
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset ROS_PACKAGE_PATH
unset PYTHONPATH
unset LD_LIBRARY_PATH

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-ros

# Source ROS2
source /opt/ros/humble/setup.bash

# Source ROS workspace containing franka_msgs
source ~/ros_ws/install/setup.bash

# Restore Conda precedence after ROS sourcing
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH


# Default task
TASK="test task"

# Override task from CLI argument
if [ -n "$1" ]; then
    TASK="$1"
fi

# Move to repo root
cd ~/project/lerobot_inference

# Run rollout
python3 examples/franka_fr3_logos/continuous_rollout.py \
    --action_ee \
    --fps 10 \
    --policy_type pi05 \
    --checkpoint_path ../ckpts/last/pretrained_model/ \
    --mode sync \
    --action_delta \
    --max_relative_target 0.05 \
    --fps 10 \
    --task "$TASK"