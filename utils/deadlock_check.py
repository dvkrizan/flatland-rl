import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import fast_count_nonzero


def get_agent_positions(env):
    agent_positions: np.ndarray = np.full((env.height, env.width), -1)
    for agent_handle in env.get_agent_handles():
        agent = env.agents[agent_handle]
        if agent.status == RailAgentStatus.ACTIVE:
            position = agent.position
            if position is None:
                position = agent.initial_position
            agent_positions[position] = agent_handle
    return agent_positions


def get_agent_targets(env):
    agent_targets = []
    for agent_handle in env.get_agent_handles():
        agent = env.agents[agent_handle]
        if agent.status == RailAgentStatus.ACTIVE:
            agent_targets.append(agent.target)
    return agent_targets


def check_for_deadlock(handle, env, agent_positions, check_position=None, check_direction=None):
    agent = env.agents[handle]
    if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
        return False

    position = agent.position
    if position is None:
        position = agent.initial_position
    if check_position is not None:
        position = check_position
    direction = agent.direction
    if check_direction is not None:
        direction = check_direction

    possible_transitions = env.rail.get_transitions(*position, direction)
    num_transitions = fast_count_nonzero(possible_transitions)
    for dir_loop in range(4):
        if possible_transitions[dir_loop] == 1:
            new_position = get_new_position(position, dir_loop)
            opposite_agent = agent_positions[new_position]
            if opposite_agent != handle and opposite_agent != -1:
                num_transitions -= 1
            else:
                return False

    is_deadlock = num_transitions <= 0
    return is_deadlock


def check_if_all_blocked(env):
    """
    Checks whether all the agents are blocked (full deadlock situation).
    In that case it is pointless to keep running inference as no agent will be able to move.
    :param env: current environment
    :return:
    """

    # First build a map of agents in each position
    location_has_agent = {}
    for agent in env.agents:
        if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
            location_has_agent[tuple(agent.position)] = 1

    # Looks for any agent that can still move
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            continue

        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)

                if new_position not in location_has_agent:
                    return False

    # No agent can move at all: full deadlock!
    return True
