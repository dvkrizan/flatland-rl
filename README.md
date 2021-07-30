🚂 This code is based on the official starter kit - NeurIPS 2020 Flatland Challenge
---

You can use for your own experiments full or reduced action space. 

```python
def map_action(action):
    # if full action space is used -> no mapping required
    if get_action_size() == get_flatland_full_action_size():
        return action
    
    # if reduced action space is used -> the action has to be mapped to real flatland actions
    # The reduced action space removes the DO_NOTHING action from Flatland.
    if action == 0:
        return RailEnvActions.MOVE_LEFT
    if action == 1:
        return RailEnvActions.MOVE_FORWARD
    if action == 2:
        return RailEnvActions.MOVE_RIGHT
    if action == 3:
        return RailEnvActions.STOP_MOVING
```

```python
set_action_size_full()
```
or 
```python
set_action_size_reduced()
```
action space. The reduced action space just removes DO_NOTHING. 

---
The used policy is based on the FastTreeObs in the official starter kit - NeurIPS 2020 Flatland Challenge. But the
 FastTreeObs in this repo is an extended version. 
[fast_tree_obs.py](./utils/fast_tree_obs.py)

---
Have a look into the [run.py](./run.py) file. There you can select using PPO or DDDQN as RL agents. 
 
```python
####################################################
# EVALUATION PARAMETERS
set_action_size_full()

# Print per-step logs
VERBOSE = True
USE_FAST_TREEOBS = True

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116591 adrian_egli
    # graded	71.305	0.633	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/51
    # Fri, 22 Jan 2021 23:37:56
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122120236-3000.pth"  # 17.011131341978228
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116658 adrian_egli
    # graded	73.821	0.655	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/52
    # Sat, 23 Jan 2021 07:41:35
    set_action_size_reduced()
    load_policy = "PPO"
    checkpoint = "./checkpoints/210122235754-5000.pth"  # 16.00113400887389
    EPSILON = 0.0

if True:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116659 adrian_egli
    # graded	80.579	0.715	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/53
    # Sat, 23 Jan 2021 07:45:49
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122165109-5000.pth"  # 17.993750197899438
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # !! This is not a RL solution !!!!
    # -------------------------------------------------------------------------------------------------------
    # 116727 adrian_egli
    # graded	106.786	0.768	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/54
    # Sat, 23 Jan 2021 14:31:50
    set_action_size_reduced()
    load_policy = "DeadLockAvoidance"
    checkpoint = None
    EPSILON = 0.0
```

---
A deadlock avoidance agent is implemented. The agent only lets the train take the shortest route. And it tries to avoid as many deadlocks as possible.
* [dead_lock_avoidance_agent.py](./utils/dead_lock_avoidance_agent.py)


---
The policy interface has changed, please have a look into 
* [policy.py](./reinforcement_learning/policy.py)

---
See the tensorboard training output to get some insights:
```
tensorboard --logdir ./runs_bench 
```

---
```
python reinforcement_learning/multi_agent_training.py --use_fast_tree_observation  --checkpoint_interval 1000 -n 5000
 --policy DDDQN -t 2 --action_size reduced --buffer_siz 128000
```

[multi_agent_training.py](./reinforcement_learning/multi_agent_training.py)
has new or changed parameters. Most important new or changed parameters for training. 
 * policy :  [DDDQN, PPO, DeadLockAvoidance, DeadLockAvoidanceWithDecision, MultiDecision] : Default value
   DeadLockAvoidance 
 * use_fast_tree_observation : [false,true] : Default value = true  
 * action_size: [full, reduced] : Default value = full
``` 
usage: multi_agent_training.py [-h] [-n N_EPISODES] [--n_agent_fixed]
                               [-t TRAINING_ENV_CONFIG]
                               [-e EVALUATION_ENV_CONFIG]
                               [--n_evaluation_episodes N_EVALUATION_EPISODES]
                               [--checkpoint_interval CHECKPOINT_INTERVAL]
                               [--eps_start EPS_START] [--eps_end EPS_END]
                               [--eps_decay EPS_DECAY]
                               [--buffer_size BUFFER_SIZE]
                               [--buffer_min_size BUFFER_MIN_SIZE]
                               [--restore_replay_buffer RESTORE_REPLAY_BUFFER]
                               [--save_replay_buffer SAVE_REPLAY_BUFFER]
                               [--batch_size BATCH_SIZE] [--gamma GAMMA]
                               [--tau TAU] [--learning_rate LEARNING_RATE]
                               [--hidden_size HIDDEN_SIZE]
                               [--update_every UPDATE_EVERY]
                               [--use_gpu USE_GPU] [--num_threads NUM_THREADS]
                               [--render] [--load_policy LOAD_POLICY]
                               [--use_fast_tree_observation]
                               [--max_depth MAX_DEPTH] [--policy POLICY]
                               [--action_size ACTION_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -n N_EPISODES, --n_episodes N_EPISODES
                        number of episodes to run
  --n_agent_fixed       hold the number of agent fixed
  -t TRAINING_ENV_CONFIG, --training_env_config TRAINING_ENV_CONFIG
                        training config id (eg 0 for Test_0)
  -e EVALUATION_ENV_CONFIG, --evaluation_env_config EVALUATION_ENV_CONFIG
                        evaluation config id (eg 0 for Test_0)
  --n_evaluation_episodes N_EVALUATION_EPISODES
                        number of evaluation episodes
  --checkpoint_interval CHECKPOINT_INTERVAL
                        checkpoint interval
  --eps_start EPS_START
                        max exploration
  --eps_end EPS_END     min exploration
  --eps_decay EPS_DECAY
                        exploration decay
  --buffer_size BUFFER_SIZE
                        replay buffer size
  --buffer_min_size BUFFER_MIN_SIZE
                        min buffer size to start training
  --restore_replay_buffer RESTORE_REPLAY_BUFFER
                        replay buffer to restore
  --save_replay_buffer SAVE_REPLAY_BUFFER
                        save replay buffer at each evaluation interval
  --batch_size BATCH_SIZE
                        minibatch size
  --gamma GAMMA         discount factor
  --tau TAU             soft update of target parameters
  --learning_rate LEARNING_RATE
                        learning rate
  --hidden_size HIDDEN_SIZE
                        hidden size (2 fc layers)
  --update_every UPDATE_EVERY
                        how often to update the network
  --use_gpu USE_GPU     use GPU if available
  --num_threads NUM_THREADS
                        number of threads PyTorch can use
  --render              render 1 episode in 100
  --load_policy LOAD_POLICY
                        policy filename (reference) to load
  --use_fast_tree_observation
                        use FastTreeObs instead of stock TreeObs
  --max_depth MAX_DEPTH
                        max depth
  --policy POLICY       policy name [DDDQN, PPO, DeadLockAvoidance,
                        DeadLockAvoidanceWithDecision, MultiDecision]
  --action_size ACTION_SIZE
                        define the action size [reduced,full]
```                        


---
If you have any questions write me on the official discord channel **aiAdrian**    
(Adrian Egli - adrian.egli@gmail.com) 


Credits
---

* Florian Laurent <florian@aicrowd.com>
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Sharada Mohanty <mohanty@aicrowd.com>
* Christian Baumberger <christian.baumberger@sbb.ch>
* Guillaume Mollard <guillaume.mollard2@gmail.com>

Main links
---

* [Flatland documentation](https://flatland.aicrowd.com/)
* [Flatland Challenge](https://www.aicrowd.com/challenges/flatland)

Communication
---

* [Discord Channel](https://discord.com/invite/hCR3CZG)
* [Discussion Forum](https://discourse.aicrowd.com/c/neurips-2020-flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)