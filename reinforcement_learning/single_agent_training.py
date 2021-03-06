import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path
import os
import shutil

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.ppo_agent import PPOPolicy
import matplotlib.pyplot as plt
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

"""
This file shows how to train a single agent using a reinforcement learning approach.
Documentation: https://flatland.aicrowd.com/getting-started/rl/single-agent.html

This is a simple method used for demonstration purposes.
multi_agent_training.py is a better starting point to train your own solution!
"""


def train_agent(n_episodes):
    # Environment parameters
    n_agents = 1
    x_dim = 25
    y_dim = 25
    n_cities = 2
    max_rails_between_cities = 2
    max_rail_pairs_in_city = 2
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
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
    )
    
    env.reset(True, True)
    env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=1000,
                          screen_width=1000)

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    print('n_features_per_node', n_features_per_node)
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    #max_steps = 20
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_dict = dict()

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)  # todo smooth when rendering instead
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
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
        'use_gpu': True
    }

    # Double Dueling DQN policy
    #policy = DDDQNPolicy(state_size, action_size, Namespace(**training_parameters))
    
    # PPO policy
    policy = PPOPolicy(state_size, action_size, in_parameters=Namespace(**training_parameters))

    for episode_idx in range(n_episodes):
        print('episode #', episode_idx)
        if episode_idx % 100 == 0:
            dirName = 'images/episode_{}'.format(episode_idx)
            if os.path.exists(dirName):
                shutil.rmtree(dirName)
            os.mkdir(dirName)
        
        score = 0

        # Reset environment
        #print('Initial position 1', env.agents[0].initial_position)
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        print('Initial position', env.agents[0].initial_position)
        print('Target', env.agents[0].target)
        
        env_renderer.reset()

        # Build agent specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth,
                                                         observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()
            #print('agent', agent, 'observation', agent_obs[agent])
        
        frame_step = 0
        # Run episode
        for step in range(max_steps - 1):
            if step % 100 == 0:
                print(f'episode', episode_idx, 'step #', step)
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    update_values = True
                    action = policy.act(agent, agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            if episode_idx % 100 == 0:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
                env_renderer.gl.save_image("./Images/episode_{}/flatland_frame_{:04d}.png".format(episode_idx, frame_step))
            frame_step += 1

            # Update replay buffer and train agent
            for agent in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[agent]:
                    policy.step(agent,
                                agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                                agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth,
                                                             observation_radius=10)

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
        action_probs = action_count / np.sum(action_count)

        if episode_idx % 100 == 0:
            end = "\n"
            policy.save('./checkpoints/single-' + str(episode_idx) + '.pth')
            action_count = [1] * action_size
        else:
            end = " "

        print(
            '\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(),
                x_dim, y_dim,
                episode_idx,
                np.mean(scores_window),
                100 * np.mean(completion_window),
                eps_start,
                action_probs
            ), end=end)

    # Plot overall training progress at the end
    #plt.plot(scores)
    #plt.show()

    #plt.plot(completion)
    #plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", dest="n_episodes", help="number of episodes to run", default=500,
                        type=int)
    args = parser.parse_args()

    train_agent(args.n_episodes)
