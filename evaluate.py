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
from sb3_contrib import QRDQN
from gymnasium.spaces import Discrete

# diambra run -r "$PWD/roms/" python evaluate.py --cfgFile "$PWD/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"


def main(cfg_file):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

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

    # Create environment
    env, num_envs = make_sb3_env(
        settings.game_id, settings, wrappers_settings, render_mode="human"
    )
    print("Activated {} environment(s)".format(num_envs))

    # PPO settings
    ppo_settings = params["ppo_settings"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    # Load the trained agent
    # env.action_space = Discrete(18)
    agent = QRDQN.load(
        "./QRDQN_5mil_steps.zip",
        env=env,
    )

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Run trained agent
    observation = env.reset()
    cumulative_reward = 0
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        cumulative_reward += reward
        if reward != 0:
            print("Cumulative reward =", cumulative_reward)

        if done:
            observation = env.reset()
            break

    # Close the environment
    env.close()

    # Return success
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)
