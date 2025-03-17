import csv
import os
import yaml
import json
import argparse
from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.make_sb3_env import (
    make_sb3_env,
    EnvironmentSettings,
    WrappersSettings,
)
from sb3_contrib import QRDQN

"""This is an example agent based on stable baselines 3.

Usage:
diambra run python stable_baselines3/agent.py --cfgFile $PWD/stable_baselines3/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""


def main(cfg_file, trained_model, test=False):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    # model_folder = os.path.join(
    #     base_path,
    #     params["folders"]["parent_dir"],
    #     params["settings"]["game_id"],
    #     params["folders"]["model_name"],
    #     "model",
    # )
    model_folder = os.path.join(
        base_path,
        params["folders"]["parent_dir"],
    )

    # Settings
    params["settings"]["action_space"] = (
        SpaceTypes.DISCRETE
        if params["settings"]["action_space"] == "discrete"
        else SpaceTypes.MULTI_DISCRETE
    )
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(
        WrappersSettings, params["wrappers_settings"]
    )
    wrappers_settings.normalize_reward = False

    # Create environment
    # env, num_envs = make_sb3_env(
    #     settings.game_id, settings, wrappers_settings, render_mode="human", no_vec=True
    # )
    env, num_envs = make_sb3_env(
        settings.game_id, settings, wrappers_settings, render_mode="human"
    )
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
    model_path = os.path.join(model_folder, trained_model)
    agent = QRDQN.load(model_path, env=env)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create CSV to save to
    num_iterations = 30
    if params["episode_saving_settings"]["num_iters"] and isinstance(
        params["episode_saving_settings"]["num_iters"], int
    ):
        num_iterations = params["episode_saving_settings"]["num_iters"]
    highest_stage_completed = 0
    cumulative_stage_reached = 0
    data = [["Highest Stage:"], ["Avg Stage:"], ["Iteration Number", "Stage Completed"]]

    if not params["episode_saving_settings"]["csv_filename"]:
        print("CSV Filename not found")
        return 1
    else:
        filename = params["episode_saving_settings"]["csv_filename"]

    current_iter = 1

    # obs, info = env.reset()
    observation = env.reset()
    cumulative_reward = 0
    while True and num_iterations >= current_iter:
        env.render()

        # action, _ = agent.predict(obs, deterministic=True)
        action, _ = agent.predict(observation, deterministic=False)

        # obs, reward, terminated, truncated, info = env.step(action.tolist())
        observation, reward, done, info = env.step(action)

        cumulative_reward += reward
        if reward != 0:
            # print("Cumulative reward =", cumulative_reward)
            pass
        # print(f"Observation = {observation}")
        # print(f"Action = {action}")

        # if terminated or truncated:
        if done:

            print(f"ITERATION {current_iter} / {num_iterations}")

            stage_reached = get_stage(info[0]["terminal_observation"]["stage"][0])
            print(f"STAGE REACHED: {stage_reached}")

            if stage_reached > highest_stage_completed:
                highest_stage_completed = stage_reached

            cumulative_stage_reached += stage_reached

            data.append([current_iter, stage_reached])
            current_iter += 1

            # print("-----------------------INFO-------------------")
            # print(f"{info[0].keys()}")
            # info["stage"]

            # print("---------------------------------------------------------------------------"
            # "\n\t\t\t\t\t\OBSERVATION\n"
            # "---------------------------------------------------------------------------")
            # print(observation)

            # observation, info = env.reset()
            observation = env.reset()
            if test is True:
                break
            # if info["env_done"] or test is True:
            #     break

    # Close the environment
    env.close()

    # Save highest and average stage
    data[0].append(highest_stage_completed)  # highest
    data[1].append(cumulative_stage_reached / num_iterations)  # average

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Return success
    return 0


# Given the float stage val, return the integer stage
def get_stage(raw_stage: float) -> int:
    stage = int(raw_stage * 10)
    return stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument(
        "--trainedModel", type=str, default="model", help="Model checkpoint"
    )
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel, bool(opt.test))
