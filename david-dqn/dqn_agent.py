import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer


class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
    mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
    replace_target_steps=1000, algo=None, env_name=None, chkpt_dir='checkpoints/'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_steps = replace_target_steps
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + "_" + self.algo + "_q_eval",
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + "_" + self.algo + "_q_next",
                                   chkpt_dir=self.chkpt_dir)


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # pytorch needs an input of shape batch size, input dims (which here is 4, 84, 84)
            # We can add the batch size of 1 by wrapping observation in a list
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            # Return action as the highest q value of the predicted actions
            # Note we are returning an integer of the action argument in the action list
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action


    def store_transition(self, state, action, reward, state_, done):
        # Save state, action, reward, new state transition to mmory
        self.memory.store_transition(state, action, reward, state_, done)


    def sample_memory(self):
        # Uniformly sample memory
        state, action, reward, new_state, done = \
                            self.memory.sample_buffer(self.batch_size)
        # Convert to pytorch tensors
        # Using T.tensor maintains the data type of the underly numpy array
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones


    def replace_target_network(self):
        # Only update q_next every self.replace_target_steps
        if self.learn_step_counter % self.replace_target_steps == 0:
            # Load the cnn state dictionary of q_eval onto q_next
            self.q_next.load_state_dict(self.q_eval.state_dict())


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min


    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()


    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


    def learn(self):
        # Don't start learning until memory counter is >= batch size (which we set at 32)
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # Zero grad on the q_eval cnn
        # Note we aren't backpropgating the q_pred network so no need to zero_grad that cnn
        self.q_eval.optimizer.zero_grad()
        
        # Load the cnn state dictionary of q_eval onto q_next. however
        # recal we only do this every replace_target_steps (which we set at 1000 steps)
        self.replace_target_network()

        # Sample memory
        states, actions, rewards, states_, dones = self.sample_memory()
       
        ## Get q_pred from action taken (which is taken from sample memory above)
        # Note we have batch size x actions (32 x 6 for now) so we need to make a 
        # numpy array of the batch indices
        indices = np.arange(self.batch_size)
        # Predict state action value q with states as input (again we are running a batch of states)
        # q_pred returns a 2D array of shape batch size x number actions
        q_pred = self.q_eval.forward(states)
        # Find the q_pred of the action taken in each of the frames of the batch
        q_pred = q_pred[indices, actions]
        
        ## Predict q_next with state_ as input
        # This will return a q value for each action for each of the frames in the batch
        # i.e. shape batch size x number of actions
        q_next = self.q_next.forward(states_)
        # Now we went the max q value for each frame in the batch
        # Note pytorch max function returns a tuple of max value and it's indice,
        # so we need to request [0] to just get the max values (for each frame in the batch)
        q_next = q_next.max(dim=1)[0]

        # Use boolean masking to set q_next = 0 wherever done=True or 1 in the batch
        # This is by our definition that the q value of the terminal state is 0
        q_next[dones] = 0.0
        # Calculate q_target for each frame in the batch
        # Note if that it is in the terminal state, q_target will be equal to rewards only
        q_target = rewards + self.gamma * q_next
        # Calculate loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        # Propogate loss backwards through cnn
        loss.backward()
        # Update cnn network weights and biases
        self.q_eval.optimizer.step()
        
        self.learn_step_counter += 1
        self.decrement_epsilon()

    