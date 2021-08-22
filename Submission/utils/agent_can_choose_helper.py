from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import fast_count_nonzero


class AgentCanChooseHelper:
    def __init__(self):
        pass

    def build_data(self, env):
        self.env = env
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()

    def find_all_switches(self):
        # Search the environment (rail grid) for all switch cells. A switch is a cell where more than one tranisation
        # exists and collect all direction where the switch is a switch.
        self.switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        if pos not in self.switches.keys():
                            self.switches.update({pos: [dir]})
                        else:
                            self.switches[pos].append(dir)

    def find_all_switch_neighbours(self):
        # Collect all cells where is a neighbour to a switch cell. All cells are neighbour where the agent can make
        # just one step and he stands on a switch. A switch is a cell where the agents has more than one transition.
        self.switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                # look one step forward
                for dir in range(4):
                    pos = (h, w)
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in self.switches.keys() and pos not in self.switches.keys():
                                if pos not in self.switches_neighbours.keys():
                                    self.switches_neighbours.update({pos: [dir]})
                                else:
                                    self.switches_neighbours[pos].append(dir)

    def find_all_cell_where_agent_can_choose(self):
        # prepare the memory - collect all cells where the agent can choose more than FORWARD/STOP.
        self.find_all_switches()
        self.find_all_switch_neighbours()

    def check_agent_decision(self, position, direction):
        # Decide whether the agent is
        # - on a switch
        # - at a switch neighbour (near to switch). The switch must be a switch where the agent has more option than
        #   FORWARD/STOP
        # - all switch : doesn't matter whether the agent has more options than FORWARD/STOP
        # - all switch neightbors : doesn't matter the agent has more then one options (transistion) when he reach the
        #   switch
        agents_on_switch = False
        agents_on_switch_all = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in self.switches.keys():
            agents_on_switch = direction in self.switches[position]
            agents_on_switch_all = True

        if position in self.switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in self.switches.keys():
                if not direction in self.switches[new_cell]:
                    agents_near_to_switch = direction in self.switches_neighbours[position]
            else:
                agents_near_to_switch = direction in self.switches_neighbours[position]

            agents_near_to_switch_all = direction in self.switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def required_agent_decision(self):
        agents_can_choose = {}
        agents_on_switch = {}
        agents_on_switch_all = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all, ret_agents_on_switch_all = \
                self.check_agent_decision(
                    self.env.agents[a].position,
                    self.env.agents[a].direction)
            agents_on_switch.update({a: ret_agents_on_switch})
            agents_on_switch_all.update({a: ret_agents_on_switch_all})
            ready_to_depart = self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch and not ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all and not ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all
