export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/descfly/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/torch/lib

${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/play.py \
    --num_envs 10 \
    --robot x2t2 \
    --reference_motion_path neural_wbc/data/data/x2_t2_23dof_retarget_motion_dynamic_adjust \
    --teacher_policy.resume_path logs/teacher/25_10_31_21-45-20/ \
    --teacher_policy.checkpoint model_20000.pt