import argparse
import os
import pickle
from importlib import metadata
from pathlib import Path
from datetime import datetime
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env_rough_rl import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, required=True, help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes")
    parser.add_argument("--show-viewer", action="store_true", help="Show Genesis viewer")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        add_eval_camera=True,
        show_viewer=True,
    )

    # 評価用の状態記録を有効化
    env.record_eval_metrics = True

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(
        os.path.join(log_dir, f"model_{args.ckpt}.pt"),
        map_location="cpu"
        )
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()

    # 録画用カメラを取得
    camera = env.eval_camera

    # Go2を追従するように設定
    camera.follow_entity(
        env.robot,
        fixed_axis=(None, None, 1.6),
        smoothing=0.1,
        fix_orientation=False,
        )

    # 評価結果を保存するリスト
    episode_distances = []
    episode_displacements = []
    episode_durations = []

    falls = 0
    timeouts = 0
    other_terminations = 0

    # 最初の位置
    prev_pos = env.base_pos[0].detach().cpu().clone()
    episode_start_pos = prev_pos.clone()

    # 現在のエピソードの累積走破距離
    episode_distance = 0.0

    roll_limit = float(
        env.env_cfg["termination_if_roll_greater_than"]
    )

    pitch_limit = float(
        env.env_cfg["termination_if_pitch_greater_than"]
    )

    print(f"Experiment name：{args.exp_name}")
    print(f"CheckPoint：{args.ckpt}")
    print(f"Episodes：{args.episodes}")
    print("-" * 50)

    # ==========================================
    # Video Recording Settings
    # ==========================================

    # 保存先：go2_eval_rough.py と同じディレクトリ内の videos
    video_dir = Path(__file__).resolve().parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名に日時を付加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_path = video_dir / f"go2_eval_{timestamp}.mp4"

    # 録画FPS
    target_fps = 30.0

    # シミュレーションのdtに合わせて実現可能なFPSを計算
    steps_per_frame = max(
        1,
        round(1.0 / (target_fps * env.dt))
        )

    record_fps = 1.0 / (steps_per_frame * env.dt)

    # 録画開始
    camera.start_recording(
        save_to_filename=str(video_path),
        fps=record_fps
        )

    print("\n===== Video Recording =====")
    print(f"Save path: {video_path}")
    print(f"Recording FPS: {record_fps:.2f}")

    try:
        step_count = 0

        with torch.no_grad():
            while len(episode_distances) < args.episodes:
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)

                step_count += 1

                # 100ステップごとに報酬寄与を表示
                if step_count % 100 == 0:
                    print(f"\n===== Step {step_count} =====")
                    for name, value in env.reward_terms.items():
                        contribution = value[0].item()
                        print(
                            f"{name:25s}: {contribution:+.6f}"
                            )
                    print(
                        f"{'Total reward':25s}: {env.rew_buf[0].item():+.6f}"
                        )

                # リセット前の位置を取得
                pos = (
                    infos["eval_base_pos"][0]
                    .detach()
                    .cpu()
                    )

                # 水平方向の移動距離を加算
                step_distance = torch.linalg.vector_norm(
                    pos[:2] - prev_pos[:2]
                ).item()

                episode_distance += step_distance
                prev_pos = pos.clone()

                # エピソードが終了していなければ次のステップへ
                if not bool(dones[0].item()):
                    continue
            
                # リセット前の姿勢を取得
                euler = (
                    infos["eval_base_euler"][0]
                    .detach()
                    .cpu()
                    )

                # ロール角・ピッチ角
                roll = abs(euler[0].item())
                pitch = abs(euler[1].item())

                # エピソードの長さを秒に変換
                episode_steps = int(
                    infos["eval_episode_length"][0].item()
                    )

                duration = episode_steps * env.dt

                # 制限時間まで到達したか
                timeout = bool(infos["time_outs"][0].item())

                # 環境の終了条件に基づいて転倒を判定
                fell = (
                    roll > roll_limit
                    or pitch > pitch_limit
                    )

                # エピソードの分類
                if fell:
                    falls += 1
                    status = "Fall down"

                elif timeout:
                    timeouts += 1
                    status = "Completed Walking"

                else:
                    other_terminations += 1
                    status = "Error"

                # 走破距離・歩行時間
                episode_distances.append(episode_distance)
                episode_durations.append(duration)

                # 開始位置から終了位置までの水平変位
                displacement = torch.linalg.vector_norm(
                    pos[:2] - episode_start_pos[:2]
                    ).item()

                episode_displacements.append(displacement)

                episode_number = len(episode_distances)

                print(
                    f"Episode {episode_number:3d} | "
                    f"Result: {status:10s} | "
                    f"Time: {duration:6.2f} s | "
                    f"Distance: {episode_distance:7.3f} m"
                )

                print("Reward scales:", env.reward_scales)
                print("Reward functions:", env.reward_functions.keys())

                
                print("\n===== Episode Reward Contributions =====")
                contributions = infos.get("episode_reward_contributions")
                if contributions is not None:
                    total_reward = 0.0
                    for name, value in contributions.items():
                        contribution = value[0].item()
                        total_reward += contribution
                        print(f"{name:25s}: {contribution:+.6f}")

                    print("-" * 40)
                    print(f"{'Total reward':25s}: {total_reward:+.6f}")

                else:
                    print("Episode reward contributions were not found.")
                
                # 次のエピソードに向けて初期化
                for value in env.episode_reward_terms.values():
                    value.zero_()

                # 次のエピソードの開始位置
                # env.step()内で環境は既にリセットされている
                prev_pos = (
                    env.base_pos[0]
                    .detach()
                    .cpu()
                    .clone()
                    )

                episode_start_pos = prev_pos.clone()
                episode_distance = 0.0

        # 評価結果を集計
        total_episodes = len(episode_distances)

        fall_rate = 100.0 * falls / total_episodes
        completion_rate = 100.0 * timeouts / total_episodes
        other_rate = 100.0 * other_terminations / total_episodes

        mean_distance = sum(episode_distances) / total_episodes

        mean_displacement = (
            sum(episode_displacements) / total_episodes
        )

        mean_duration = sum(episode_durations) / total_episodes

        total_duration = sum(episode_durations)

        if total_duration > 0:
            mean_speed = sum(episode_distances) / total_duration
        else:
            mean_speed = 0.0

        # 結果を表示
        print("\n")
        print("=" * 50)
        print("        Evaluation Results of Policy")
        print("=" * 50)

        print(f"Number of Episodes      : {total_episodes}")
        print(f"Number of Fall down              : {falls}")
        print(f"Fall down rate                : {fall_rate:.2f} %")
        print(f"Completion rate  : {completion_rate:.2f} %")
        print(f"Error rate    : {other_rate:.2f} %")

        print("-" * 50)

        print(f"Mean Distance          : {mean_distance:.3f} m")
        print(f"Mean Displacement          : {mean_displacement:.3f} m")
        print(f"Mean Duration          : {mean_duration:.3f} s")
        print(f"Mean Speed          : {mean_speed:.3f} m/s")

        print("=" * 50)

    finally:
        print("Stopped")
        # 録画停止
        camera.stop_recording()

        print("\n===== Recording Finished =====")
        print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
