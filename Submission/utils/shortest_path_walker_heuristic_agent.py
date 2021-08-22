import numpy as np
from flatland.envs.rail_env import RailEnvActions

from reinforcement_learning.policy import Policy


class ShortestPathWalkerHeuristicPolicy(Policy):
    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, handle, node, eps=0.):

        left_node = node.childs.get('L')
        forward_node = node.childs.get('F')
        right_node = node.childs.get('R')

        dist_map = np.zeros(5)
        dist_map[RailEnvActions.DO_NOTHING] = np.inf
        dist_map[RailEnvActions.STOP_MOVING] = 100000
        # left
        if left_node == -np.inf:
            dist_map[RailEnvActions.MOVE_LEFT] = np.inf
        else:
            if left_node.num_agents_opposite_direction == 0:
                dist_map[RailEnvActions.MOVE_LEFT] = left_node.dist_min_to_target
            else:
                dist_map[RailEnvActions.MOVE_LEFT] = np.inf
        # forward
        if forward_node == -np.inf:
            dist_map[RailEnvActions.MOVE_FORWARD] = np.inf
        else:
            if forward_node.num_agents_opposite_direction == 0:
                dist_map[RailEnvActions.MOVE_FORWARD] = forward_node.dist_min_to_target
            else:
                dist_map[RailEnvActions.MOVE_FORWARD] = np.inf
        # right
        if right_node == -np.inf:
            dist_map[RailEnvActions.MOVE_RIGHT] = np.inf
        else:
            if right_node.num_agents_opposite_direction == 0:
                dist_map[RailEnvActions.MOVE_RIGHT] = right_node.dist_min_to_target
            else:
                dist_map[RailEnvActions.MOVE_RIGHT] = np.inf
        return np.argmin(dist_map)

    def save(self, filename):
        pass

    def load(self, filename):
        pass


policy = ShortestPathWalkerHeuristicPolicy()


def normalize_observation(observation, tree_depth: int, observation_radius=0):
    return observation
