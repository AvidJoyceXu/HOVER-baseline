#!/usr/bin/env python
import torch

from isaaclab.app import AppLauncher

def print_diff(val1, val2, name, tol=1e-4):
    err = (val1 - val2).abs().max().item()
    if err > tol:
        print(f"[DIFF][{name}] max_abs_diff = {err:.6g}")
    else:
        print(f"[PASS][{name}] aligned, max_abs_diff = {err:.3g}")


def main():
    # --- 必须先初始化 simulation_app ---
    app_launcher = AppLauncher()
    simulation_app = app_launcher.app
    # --- 必须在 simulation_app 创建后, 才能实例化 env & 相关 asset ---
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_x2t2 import NeuralWBCEnvCfgX2T2
    from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = NeuralWBCEnvCfgX2T2()
    env = NeuralWBCEnv(env_cfg, render_mode=None)

    env.reset()
    num_envs = env.num_envs

    # 加载reference motion
    ref_cfg = ReferenceMotionManagerCfg()
    ref_cfg.motion_path = env_cfg.reference_motion_manager.motion_path
    ref_cfg.skeleton_path = env_cfg.reference_motion_manager.skeleton_path
    ref_mgr = ReferenceMotionManager(ref_cfg, device=device, num_envs=num_envs, random_sample=False, extend_head=False, dt=env_cfg.dt)
    episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
    ref_state = ref_mgr.get_state_from_motion_lib_cache(
        episode_length_buf=episode_length_buf,
        quaternion_is_xyzw=True,
    )

    # 关节/身体名字与ID对齐
    robot_joint_names = env._joint_names
    robot_joint_ids = env._joint_ids
    robot_body_names = env._body_names
    robot_body_ids = env._body_ids

    # 获得motion里的joint/body名字顺序
    motion_joint_names = env_cfg.joint_names
    motion_body_names = env_cfg.body_names

    # 获取参考动作，按robot顺序重排
    ref_joint_pos = ref_state.joint_pos[0]
    ref_root_pos = ref_state.root_pos[0]
    ref_root_rot = ref_state.root_rot[0]
    # 对root rot统一wxyz
    ref_root_pose = torch.cat([ref_root_pos, ref_root_rot]).to(device)

    
    # 写入robot
    env._robot.write_root_pose_to_sim(ref_root_pose.unsqueeze(0), env_ids=torch.tensor([0]).to(device))
    env._robot.write_joint_state_to_sim(ref_joint_pos.unsqueeze(0), torch.zeros_like(ref_joint_pos).unsqueeze(0), robot_joint_ids, env_ids=torch.tensor([0]).to(device))
    # 读出robot当前状态
    sim_root_pose = env._robot.data.root_state_w
    sim_joint_pos = env._robot.data.joint_pos

    # 比较root pos
    print_diff(sim_root_pose[0][:3], ref_root_pos, 'root_pos')
    # 比较root rot
    print_diff(sim_root_pose[0][3:7], ref_root_rot, 'root_rot (wxyz)')
    # 比较joint pos
    print_diff(sim_joint_pos[0][robot_joint_ids], ref_joint_pos, 'joint_pos')

    print("Done. Check the above for any misalignment/joint mapping bug.")

if __name__ == "__main__":
    main()
    print("Done.")
