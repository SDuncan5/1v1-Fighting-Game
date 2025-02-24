import os
import yaml
import json
import argparse
import numpy as np
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import (
    make_sb3_env,
    EnvironmentSettings,
    WrappersSettings,
)
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from sb3_contrib import QRDQN

from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.spaces import Discrete
from gymnasium.spaces.space import MaskNDArray
from random import choice

# diambra run -s 8 -r "$PWD/roms/" python training.py --cfgFile "$PWD/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"

class CustomSpace(Discrete):
    def __init__(self, list : list[int]):
        self.list = list
        super().__init__(len(list))
    
    def sample(self, mask: MaskNDArray | None = None, 
               probability: MaskNDArray | None = None) -> np.int64:
        return np.int64(choice(self.list) + 1)
    
    def contains(self, x: any) -> bool:
        return x in self.list
    
    def __repr__(self) -> str:
        return f"CustomSpace({self.list})"


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
    tensor_board_folder = os.path.join(
        base_path,
        params["folders"]["parent_dir"],
        params["settings"]["game_id"],
        params["folders"]["model_name"],
        "tb",
    )

    os.makedirs(model_folder, exist_ok=True)

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
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(
        ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1]
    )
    clip_range = linear_schedule(
        ppo_settings["clip_range"][0], ppo_settings["clip_range"][1]
    )
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    # change available actions
    # env.action_space = CustomSpace([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    # env.action_space = Discrete(18)
    # print(env.action_space)

    policy_kwargs = dict(n_quantiles=50)
    agent = QRDQN(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=10000,
        gamma=gamma,
        batch_size=batch_size,
        tensorboard_log=tensor_board_folder,
        verbose=1
    )

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(
        check_freq=autosave_freq,
        num_envs=num_envs,
        save_path=model_folder,
        filename_prefix=model_checkpoint + "_QR-DQN_CL_",
    )

    # Train the agent
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    new_model_checkpoint = "QR-DQN_CL_" + str(int(model_checkpoint) + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

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
