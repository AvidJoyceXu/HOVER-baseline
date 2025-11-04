export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/descfly/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/torch/lib
CUDA_LAUNCH_BLOCKING=1 ${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 4 \
    --robot x2t2 \
    --reference_motion_path neural_wbc/data/data/x2_t2_23dof_retarget_motion_dynamic_adjust \
    # --headless
    # --reference_motion_path neural_wbc/data/data/motions