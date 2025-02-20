import os
import yaml
import json
import argparse

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import (
    make_sb3_env,
    EnvironmentSettings,
    WrappersSettings,
)
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

# R-PPO
from sb3_contrib.ppo_recurrent import RecurrentPPO


from stable_baselines3.common.vec_env import VecVideoRecorder
import numpy as np


def main(cfg_file, model_file):
    # Read cfg
    with open(cfg_file, "r") as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(
        base_path,
        params["folders"]["parent_dir"],
        params["settings"]["game_id"],
        params["folders"]["model_name"],
        "model",
    )

    # Settings
    params["settings"]["action_space"] = (
        SpaceTypes.DISCRETE
        if params["settings"]["action_space"] == "discrete"
        else SpaceTypes.MULTI_DISCRETE
    )
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(
        WrappersSettings, params["wrappers_settings"]
    )

    #  "rgb_array"
    env, num_envs = make_sb3_env(
        settings.game_id, settings, wrappers_settings, render_mode="rgb_array"
    )
    # env, num_envs = make_sb3_env(
    # settings.game_id,
    # settings,
    # wrappers_settings,
    # render_mode="rgb_array",
    # n_env=1
    # )
    print("Activated {} environment(s)".format(num_envs))

    video_folder = "./videos_eval"
    os.makedirs(video_folder, exist_ok=True)

    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=3000,
        name_prefix="evaluation",
    )

    # R-PPO model
    model_path = os.path.join(model_folder, model_file)
    print(f"Loading model from: {model_path}")
    agent = RecurrentPPO.load(model_path, env=env)

    print("Policy architecture:")
    print(agent.policy)

    obs = env.reset()
    done = [False]
    state = None
    cumulative_reward = 0.0

    while not done[0]:
        action, state = agent.predict(
            obs, state=state, episode_start=done, deterministic=True
        )
        obs, rewards, done, infos = env.step(action)

        cumulative_reward += rewards[0]

    print("Episode done, total reward:", cumulative_reward)

    env.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument(
        "--model", type=str, default="1000000.zip", help="Model file to load"
    )
    opt = parser.parse_args()
    print(opt)
    main(opt.cfgFile, opt.model)
