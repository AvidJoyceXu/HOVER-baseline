CUDA_LAUNCH_BLOCKING=1 ${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 4096 \
    --robot x2t2 \
    --headless \
    --reference_motion_path neural_wbc/data/data/x2_t2_23dof_retarget_motion_dynamic_adjust \
    --teacher_policy.resume \
    --teacher_policy.resume_path logs/teacher/25_10_29_00-15-50/ \
    --teacher_policy.checkpoint model_40000.pt \
    # --headless \
    # --reference_motion_path neural_wbc/data/data/motions