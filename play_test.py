#!/usr/bin/env python3
import os
import yaml
import json
import argparse
import numpy as np
import pygame
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent
from stable_baselines3 import PPO
import cv2
from collections import OrderedDict


def preprocess_frame(frame):
    # Convert from RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize the frame to 128x128
    resized = cv2.resize(gray, (128, 128))
    # Stack the frame 4 times (shape becomes (4, 128, 128))
    stacked = np.stack([resized] * 4, axis=0)
    # Rearrange to shape (1, 128, 128, 4) for batch and channel-last
    stacked = np.transpose(stacked, (1, 2, 0))  # now (128,128,4)
    stacked = np.expand_dims(stacked, axis=0)  # now (1,128,128,4)
    return stacked


def convert_character(char_val, num_characters=20):
    """
    Convert a character value (either a string like "18 / Ken" or an integer)
    into a one-hot encoded vector of shape (1, num_characters).
    """
    one_hot = np.zeros((1, num_characters), dtype=np.int8)
    try:
        if isinstance(char_val, str):
            char_id = int(char_val.split("/")[0].strip())
        else:
            char_id = int(char_val)
    except Exception:
        char_id = 0
    if 0 <= char_id < num_characters:
        one_hot[0, char_id] = 1
    return one_hot


def get_human_action():
    keys = pygame.key.get_pressed()

    # Movement (0-8)
    move_index = 0
    if keys[pygame.K_LEFT] and keys[pygame.K_UP]:
        move_index = 2  # UpLeft
    elif keys[pygame.K_RIGHT] and keys[pygame.K_UP]:
        move_index = 4  # UpRight
    elif keys[pygame.K_LEFT] and keys[pygame.K_DOWN]:
        move_index = 8  # DownLeft
    elif keys[pygame.K_RIGHT] and keys[pygame.K_DOWN]:
        move_index = 6  # DownRight
    elif keys[pygame.K_LEFT]:
        move_index = 1  # Left
    elif keys[pygame.K_RIGHT]:
        move_index = 5  # Right
    elif keys[pygame.K_UP]:
        move_index = 3  # Up
    elif keys[pygame.K_DOWN]:
        move_index = 7  # Down
    else:
        move_index = 0  # NoMove

    # Button (9-17) - Only first pressed button counts
    button_index = None
    if keys[pygame.K_z]:
        button_index = 9  # But0 (e.g., Light Punch)
    elif keys[pygame.K_x]:
        button_index = 10  # But1 (e.g., Heavy Punch)
    elif keys[pygame.K_c]:
        button_index = 11  # But2 (e.g., Light Kick)
    elif keys[pygame.K_v]:
        button_index = 12  # But3 (e.g., Heavy Kick)
    elif keys[pygame.K_b]:
        button_index = 13  # But4 (Special)
    elif keys[pygame.K_n]:
        button_index = 14  # But5
    elif keys[pygame.K_m]:
        button_index = 15  # But6
    elif keys[pygame.K_COMMA]:
        button_index = 16  # But7
    elif keys[pygame.K_PERIOD]:
        button_index = 17  # But8

    # If a button is pressed, override movement with a button action.
    return button_index if button_index is not None else move_index


def main(cfg_file):
    # Load configuration parameters.
    with open(cfg_file) as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(
        base_path,
        params["folders"]["parent_dir"],
        params["settings"]["game_id"],
        params["folders"]["model_name"],
        "model",
    )

    # Set up multi-agent settings.
    settings = EnvironmentSettingsMultiAgent()
    settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
    settings.characters = ("Ryu", "Ryu")
    settings.outfits = (2, 2)

    # Use diambra.arena.make (not the vectorized make_sb3_env) for multi-agent play.
    game_id = params["settings"]["game_id"]
    env = diambra.arena.make(game_id, settings, render_mode="human")

    # Load the pretrained agent for player 2.
    ppo_settings = params["ppo_settings"]
    model_checkpoint = ppo_settings["model_checkpoint"]
    agent = PPO.load(os.path.join(model_folder, model_checkpoint))

    # Initialize pygame.
    pygame.init()
    observation, info = env.reset(seed=42)
    env.show_obs(observation)

    while True:
        # Process events so the window doesn't freeze.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return

        env.render()
        print("Action Space:", env.action_space)
        human_action = get_human_action()
        print("Human action: ", human_action)

        # Build a flat observation (obs_agent) as expected by the agent.
        obs_agent = OrderedDict()

        # Dummy "action" key: now shape (1,180) as required.
        obs_agent["action"] = np.zeros((1, 180), dtype=np.int8)

        # Shared keys.
        obs_agent["frame"] = preprocess_frame(
            observation["frame"]
        )  # shape (1,128,128,4)
        obs_agent["stage"] = np.array([observation["stage"]], dtype=np.float32)
        obs_agent["timer"] = np.array([observation["timer"]], dtype=np.float32)

        # Opponent's state from Player 1.
        obs_agent["opp_character"] = convert_character(observation["P1"]["character"])
        obs_agent["opp_health"] = np.array(
            [observation["P1"]["health"]], dtype=np.float32
        )
        obs_agent["opp_side"] = np.array([observation["P1"]["side"]])

        # Agent's own state from Player 2.
        obs_agent["own_health"] = np.array(
            [observation["P2"]["health"]], dtype=np.float32
        )
        obs_agent["own_side"] = np.array([observation["P2"]["side"]])

        # Debug: print shapes for verification.
        # print("Agent observation shapes:", {k: np.shape(v) for k, v in obs_agent.items()})

        # Predict the agent's action.
        agent_action, _ = agent.predict(obs_agent, deterministic=True)
        print(agent_action)
        if isinstance(agent_action, np.ndarray):
            agent_action = int(agent_action.item())
        actions = {"agent_0": human_action, "agent_1": agent_action}

        # print("Actions:", actions)

        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        env.show_obs(observation)
        # print("Reward:", reward, "Done:", done)

        if done:
            observation, info = env.reset()
            env.show_obs(observation)
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    main(opt.cfgFile)
