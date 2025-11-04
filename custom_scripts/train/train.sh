CUDA_LAUNCH_BLOCKING=1 ${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 4096 \
    --robot x2t2 \
    --headless \
    --reference_motion_path neural_wbc/data/data/x2_t2_23dof_retarget_motion_dynamic_adjust \
