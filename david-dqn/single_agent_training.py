import random
import os, sys
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path

from PIL import Image
from datetime import datetime

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from dddqn_policy import DDDQNPolicy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool
"""
This file shows how to train a single agent using a reinforcement learning approach.
Documentation: https://flatland.aicrowd.com/getting-started/rl/single-agent.html

This is a simple method used for demonstration purposes.
multi_agent_training.py is a better starting point to train your own solution!
"""

def train_agent(n_episodes):
    
    # Flag on whether to show video
    show_video = False
    
    # Environment parameters
    n_agents = 1
    x_dim = 25
    y_dim = 25
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 42

    # Observation parameters
    observation_tree_depth = 2
    observation_radius = 10

    # Exploration parameters
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.997  # for 2500ts

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth)

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
            
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
        
    )

    env.reset(True, True)

    ## We render the initial step and show the obsereved cells as colored boxes
    if show_video == True:
        env_renderer = RenderTool(env)
        env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=True)
    
    
    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    print (f"Number of features per node is {n_features_per_node}\n")
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_dict = dict()

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)  # todo smooth when rendering instead
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
    epsilons = []
    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = False

    # Training parameters
    training_parameters = {
        'buffer_size': int(1e5),
        'batch_size': 32,
        'update_every': 8,
        'learning_rate': 0.5e-4,
        'tau': 1e-3,
        'gamma': 0.99,
        'buffer_min_size': 0,
        'hidden_size': 256,
        'use_gpu': False
    }

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, Namespace(**training_parameters))

  
    for episode_idx in range(n_episodes):

        score = 0
        epsilons.append(eps_start)

        # Reset environment
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        # Reset rendering of video
        if show_video == True:
            env_renderer.reset()

        # Build agent specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():

                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=eps_start)

                    position = env.agents[agent].position
                    if position: # Agent is on the map
                        direction = env.agents[agent].direction
                        transitions = np.asarray(env.rail.get_transitions(*position, direction))
                        # Check if the action requested is allowed. If not return action to stand still.
                        allowable_action = policy.allowable_actions(action, position, direction, transitions)
                        action = allowable_action
                        # If an action is required, we want to store the obs at that step as well as the action
                        if action != 0:
                            update_values = True
                            action_count[action] += 1
                        else:
                            update_values = False
                            action = 0
                
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            ## Render video
            if show_video == True:
                env_renderer.render_env(show=True, frames=True, show_observations=True,
                                    show_predictions=True)
            

            # Update replay buffer and train agent
            for agent in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[agent]:
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=10)

                score += all_rewards[agent]

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collection information about training
        tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / (max_steps * env.get_num_agents()))
        completion.append((np.mean(completion_window)))
        scores.append(np.mean(scores_window))
        action_probs = np.round(action_count / np.sum(action_count), decimals=5)

        if episode_idx % 100 == 0:
            end = "\n"
            checkpoint_dir = 'david-dqn/checkpoints/'
            filename = f'single-{str(episode_idx)}.pth'
            filename_path = os.path.join(checkpoint_dir, filename)
            torch.save(policy.qnetwork_local, filename_path)
            action_count = [1] * action_size
        else:
            end = " "

        print('\rTraining {} agents on {}x{}  Episode {}  Average Score: {:.3f}  Dones: {:.2f}%  Epsilon: {:.2f}  Action Probs: {}'.format(
            env.get_num_agents(),
            x_dim, y_dim,
            episode_idx,
            np.mean(scores_window),
            100 * np.mean(completion_window),
            eps_start,
            action_probs
        ), end=end)

    # Plot overall training progress at the end
    time_now = datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss")
    fnamebase = 'david-dqn/plots/' + time_now + '_dddqn_single_agent'
    plot_training_curve(scores, "Scores", epsilons, 'Epsilons', fnamebase)
    plot_training_curve(completion, "Completions", epsilons, 'Epsilons', fnamebase)



def plot_training_curve(data1, data1label, data2, data2label, fnamebase):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, label=data1label)
    ax2 = fig.add_subplot(111, label=data2label, frame_on=False)

    # x axis is number of episodes
    x = list(range(1, len(data1) + 1, 1))

    ax.plot(x, data1, color="C0")
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel(data1label, color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2.plot(x, data2, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel(data2label, color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    fig.tight_layout() 

    filename = f'{fnamebase}_{data1label}.png' 
    plt.savefig(filename)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", dest="n_episodes", help="number of episodes to run", default=1, type=int)
    args = parser.parse_args()

    # train_agent(args.n_episodes)
    train_agent(5)