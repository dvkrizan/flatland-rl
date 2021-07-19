import gym
from gym import wrappers
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers
from datetime import datetime # just to have timestamps in the files
from PIL import Image as im # for debugging images


# Create folders
# checkpoints/ ('chkpt_dir' paremater in agent to hold cnn state dictionaries for q_pred and q_eval)
# plots/
# upload ROMS/Video Olympics -
#   Pong Sports (Paddle) (1977) (Atari, Joe Decuir - Sears) (CX2621 - 99806, 6-99806, 49-75104) ~.bin
# ! python -m atari_py.import_roms ROMS/

if __name__ == '__main__':

    env = make_env('PongNoFrameskip-v4')
    # Records epidoses
    # env = wrappers.Monitor(env, './videos/' \
    #                         + datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss") \
    #                         + '/')
    
    n_games = 1 # 500-5000

    best_score = -np.inf
    # Set load_checkpoint to true if it is desired to load cnn state_dict saved in 
    # the checkpoints folder
    # For training, load_checkpoint should be set to False
    # For evalution only, load_checkpoint should be set to True
    load_checkpoint = False
    

    # On full run choose mem_size=50000 transitions for 32 GB Computer 
    # and replace_target_steps=10000
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size = 42000,
                     eps_min=0.01, batch_size=32, replace_target_steps=1000, eps_dec=1e-5,
                     chkpt_dir='checkpoints/', algo='DQNAgent',
                     env_name='Pong-v0')

    # If running with load_checkpoint=True, load the cnn models for q_pred and q_eval
    if load_checkpoint:
        agent.load_models()

    # Filenames for plots
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + \
            '_' + str(n_games) + '_games'
    figure_file = 'plots/' \
                + datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss") + '_' \
                + fname + '.png'

    # Initalize values of steps, scores, eps_history, and steps_array
    n_steps = 0
    scores , eps_history, steps_array = [], [], []

    ## Loop through all n_games
    for i in range (n_games):
        # Reset done flag to False
        done = False
        # Initialize score of game starting at 0
        score = 0
        # Reset environement
        observation = env.reset()
        
        while not done:
            
            # Choose action based on observation of current environemnt
            action = agent.choose_action(observation)
            # Get observation of state of environment after acion taken, reward,
            # done flag, and debug information from info
            observation_, reward, done, info = env.step(action)

            # Debugging tool to print screen-shots every 100 steps
            if n_steps % 100 ==0 or done==True:
                print(f"displaying image at episode {i} step {n_steps}")
                # Scale the image intensity from 0-1 to 0-255
                img = im.fromarray(255*observation[0])
                # Scale up the image size 5X to make it easier to view
                img = img.resize((5*observation.shape[1],5*observation.shape[1]))
                img.show()

            # Update score with reward
            # if reward >0:
            #     print(f'reward {reward} on game {i} and steps {n_steps}')
            score += reward

            if not load_checkpoint:
                # If load_checkpoint=False (we're in training mdoe), save transition information
                # from current to next state along with action, reward, and done flag
                # Note we must convert done to an integer to be used as a mask in the learn
                # function of the agent
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                # Perform learning
                agent.learn()
            # Set current observation to new observation (i.e. action taken to next state)
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode', i, 'score:', score,
                'average score %.1f best score %.1f epsilon %.2f' %
                (avg_score, best_score, agent.epsilon),
                'steps ', n_steps)
        
        if avg_score > best_score:
            # Save models every time average score exceeds best score
            # if load_checkpoint=False
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    
    plot_learning_curve(steps_array, scores, eps_history, figure_file)