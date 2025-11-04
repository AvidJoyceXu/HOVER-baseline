#!/usr/bin/env python
import torch
import argparse

from isaaclab.app import AppLauncher


def print_diff(val1, val2, name, tol=1e-4):
    err = (val1 - val2).abs().max().item()
    if err > tol:
        print(f"[DIFF][{name}] max_abs_diff = {err:.6g}")
    else:
        print(f"[PASS][{name}] aligned, max_abs_diff = {err:.3g}")


def _print_diff_batch(name, sim_val, ref_val, tol=1e-4, max_list=5):
    # sim_val, ref_val: (num_envs, D)
    with torch.no_grad():
        diff = (sim_val - ref_val).abs()
        max_err_per_env = diff.max(dim=1).values
        max_err = max_err_per_env.max().item()
        bad_mask = max_err_per_env > tol
        bad_indices = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1).tolist()
        if isinstance(bad_indices, int):
            bad_indices = [bad_indices]
        if max_err > tol:
            print(f"[DIFF][{name}] max_abs_diff = {max_err:.6g}; bad_envs={bad_indices[:max_list]}")
        else:
            print(f"[PASS][{name}] aligned, max_abs_diff = {max_err:.3g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=600, help="Number of steps to play the demo")
    parser.add_argument("--xyzw_input", action="store_true", help="Treat ref motion quaternions as xyzw (no conversion) to test mismatch")
    parser.add_argument("--vis_markers", action="store_true", help="Visualize reference markers while playing")
    args = parser.parse_args()

    # --- 必须先初始化 simulation_app ---
    app_launcher = AppLauncher()
    simulation_app = app_launcher.app
    # --- 必须在 simulation_app 创建后, 才能实例化 env & 相关 asset ---
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_x2t2 import NeuralWBCEnvCfgX2T2
    from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = NeuralWBCEnvCfgX2T2()
    env = NeuralWBCEnv(env_cfg)

    env.reset()
    num_envs = env.num_envs # default to 2

    # 加载reference motion
    ref_cfg = ReferenceMotionManagerCfg()
    ref_cfg.motion_path = env_cfg.reference_motion_manager.motion_path
    ref_cfg.skeleton_path = env_cfg.reference_motion_manager.skeleton_path
    print("motion path: ", ref_cfg.motion_path)
    print("skeleton path: ", ref_cfg.skeleton_path)
    ref_mgr = ReferenceMotionManager(ref_cfg, device=device, num_envs=num_envs, random_sample=False, extend_head=False, dt=env_cfg.dt)

    # 关节/身体名字与ID对齐
    robot_joint_names = env._joint_names
    robot_joint_ids = env._joint_ids

    # 取第一帧用于对齐校验
    episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
    ref_state0 = ref_mgr.get_state_from_motion_lib_cache(
        episode_length_buf=episode_length_buf,
        quaternion_is_xyzw=(not args.xyzw_input),  # 默认做xyzw->wxyz转换；若指定xyzw_input则不转换以暴露错误
    )
    # ref_state0 = ref_mgr.get_state_from_motion_lib_cache(
    #     episode_length_buf=episode_length_buf,
    # )

    ref_joint_pos = ref_state0.joint_pos[0]
    ref_root_pos = ref_state0.root_pos[0]
    ref_root_rot = ref_state0.root_rot[0]  # 若未转换且为xyzw，将导致后续写入显错

    ref_root_pose = torch.cat([ref_root_pos, ref_root_rot]).to(device)

    # 写入并做一次diff校验
    env._robot.write_root_pose_to_sim(ref_root_pose.unsqueeze(0), env_ids=torch.tensor([0]).to(device))
    env._robot.write_joint_state_to_sim(ref_joint_pos.unsqueeze(0), torch.zeros_like(ref_joint_pos).unsqueeze(0), robot_joint_ids, env_ids=torch.tensor([0]).to(device))

    sim_root_pose = env._robot.data.root_state_w
    sim_joint_pos = env._robot.data.joint_pos

    print_diff(sim_root_pose[0][:3], ref_root_pos, 'root_pos')
    print_diff(sim_root_pose[0][3:7], ref_root_rot, 'root_rot (wxyz expected)')
    print_diff(sim_joint_pos[0][robot_joint_ids], ref_joint_pos, 'joint_pos')

    # 播放整个演示
    zero_action = torch.zeros_like(env.actions)
    for step in range(args.max_steps):
        episode_length_buf[:] = step
        ref_state = ref_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf,
            quaternion_is_xyzw=(not args.xyzw_input),
        )
        # ref_state = ref_mgr.get_state_from_motion_lib_cache(
        #     episode_length_buf=episode_length_buf,
        # )
        # 批量写入所有env的状态
        rpos = ref_state.root_pos              # (num_envs, 3)
        rrot = ref_state.root_rot              # (num_envs, 4) wxyz
        jpos = ref_state.joint_pos            # (num_envs, num_joints)

        # root_pose: [xyz, wxyz] -> (num_envs, 7)
        root_pose = torch.cat([rpos, rrot], dim=1)
        env_ids_tensor = torch.arange(env.num_envs, device=device)
        env._robot.write_root_pose_to_sim(root_pose, env_ids=env_ids_tensor)
        env._robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), robot_joint_ids, env_ids=env_ids_tensor)
        # 读取并批量对比
        sim_root_pose = env._robot.data.root_state_w[env_ids_tensor]
        sim_joint_pos = env._robot.data.joint_pos[env_ids_tensor]
        # root pos/rot
        _print_diff_batch('root_pos', sim_root_pose[:, :3], rpos)
        _print_diff_batch('root_rot(wxyz)', sim_root_pose[:, 3:7], rrot)
        # joints按 robot_joint_ids 对齐
        _print_diff_batch('joint_pos', sim_joint_pos[:, robot_joint_ids], jpos)


        if args.vis_markers:
            try:
                env._ref_motion_visualizer.visualize(ref_state)
            except Exception:
                pass

        # 推进仿真
        env.step(zero_action)

    print("Done. Played the demonstration. Toggle --xyzw_input to test quaternion convention.")

if __name__ == "__main__":
    main()
    print("Done.")
