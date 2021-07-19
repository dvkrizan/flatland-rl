import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)   
        # Use a boolean data type as we use self.terminal_memory as a boolean mask later on
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)                           

    def store_transition(self, state, action, reward, state_, done):
        # Find index to start writing memories. Use modulo self.mem_size
        # as this will reset the index to 0 if the mem_cntr exceeds mem_size
        # (i.e. it will over-write the variables below so we don't exceed the mem_size)
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Find position of last stored memory. 
        # (We want the minimum of mem_cntr or mem_size)
        max_mem = min(self.mem_cntr, self.mem_size)
        # Uniformly sample the memory batch size times, 
        # without replacement so we don't repeat memories
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones