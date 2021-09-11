import pyspiel
import numpy as np
from open_spiel.python.observation import make_observation
from open_spiel.python import rl_environment
import gym
from pettingzoo import AECEnv


class OpenSpielEnv(AECEnv):
    def __init__(self, game_name, params={}):
        open_spiel_game = pyspiel.load_game(game_name, params)
        self.game = open_spiel_game
        obs_shape = self.game.observation_tensor_size()
        self.possible_agents = list(range(2))
        self.num_actions = open_spiel_game.num_distinct_actions()
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "observation":gym.spaces.Box(shape=(obs_shape,),high=1,low=0),
                "action_mask":gym.spaces.Box(shape=(self.num_actions,),high=1,low=0),
            })
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_actions)
            for agent in self.possible_agents
        }

    def seed(self):
        # non-deterministic envs not supported
        pass

    def observe(self, agent):
        legal_actions = np.zeros_like(self.num_actions) if agent != self.agent_selection else self.state.legal_actions_mask()
        return {
            'observation': np.array(self.state.observation_tensor(agent), dtype='float32'),
            'action_mask': np.array(legal_actions, dtype="uint8"),
        }

    def reset(self):
        self.state = self.game.new_initial_state()
        self.agent_selection = self.state.current_player()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        assert not self.state.is_chance_node(), "non-deterministic envs not supported"

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        assert self.state.legal_actions_mask()[action], "action taken not legal!"

        self.state.apply_action(action)

        rewards_list = self.state.rewards()
        self.rewards = {agent: rewards_list[agent] for agent in self.agents}


        if self.state.is_terminal():
            self.dones = {agent: True for agent in self.agents}
        else:
            self.agent_selection = self.state.current_player()


        self._accumulate_rewards()

    def render(self, mode='human'):
        print(self.state)
        pass

    def close(self):
        pass


class ChangeGameEnv(AECEnv):
    '''
    The environment allows
    '''
    def __init__(self, envs, non_stationary_shift_steps=1e100, num_nonstationary_vals=5):
        self.sub_envs = envs
        self.non_stationary_shift_steps = non_stationary_shift_steps
        assert all(env.max_num_agents == 2 for env in envs)
        self.possible_agents = list(range(2))
        self.padded_obs_size = padded_obs_size = max(next(iter(env.observation_spaces.values()))['observation'].shape[0] for env in envs)
        self.padded_act_size = padded_act_size = max(next(iter(env.action_spaces.values())).n for env in envs)

        self.game_indicator_size = len(envs)
        self.noise_indicator_size = num_nonstationary_vals
        total_obs_size = padded_obs_size + self.game_indicator_size + self.noise_indicator_size
        total_act_size = padded_act_size + len(envs)
        self.num_actions = total_act_size
        self.obs_size = total_obs_size
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "observation":gym.spaces.Box(shape=(total_obs_size,),high=1,low=0),
                "action_mask":gym.spaces.Box(shape=(total_act_size,),high=1,low=0),
            })
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(total_act_size)
            for agent in self.possible_agents
        }
        self.seed()
        # These two values ARE preserved between resets,
        # sort of breaking the pettingzoo API
        self.starting_agent = 0
        self.non_stationary_steps = 0
        self.nonstationary_value = self.np_random.randint(0, self.noise_indicator_size)

    def reset(self):
        self.agent_selection = self.possible_agents[self.starting_agent]
        self.starting_agent = (self.starting_agent + 1) % len(self.possible_agents)

        self.sub_env = None
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        if self.sub_env is None:
            env_choice_start = self.padded_act_size
            env_choice = action - env_choice_start
            if env_choice < 0 or env_choice >= len(self.sub_envs):
                raise AssertionError("illegal action during choice move")
            self.sub_env = self.sub_envs[env_choice]
            self.sub_env.reset()
        else:
            self.sub_env.step(action)

        self.rewards = self.sub_env.rewards
        self._cumulative_rewards = self.sub_env._cumulative_rewards
        self.dones = self.sub_env.dones
        self.infos = self.sub_env.infos
        self.agents = self.sub_env.agents
        self.agent_selection = self.sub_env.agent_selection
        self.non_stationary_steps += 1
        if self.non_stationary_shift_steps < self.non_stationary_steps:
            self.self.np_random.randint(0, self.noise_indicator_size)
            self.non_stationary_steps = 0

    def observe(self, agent):
        if self.sub_env is None:
            action_mask = np.zeros(self.num_actions, dtype='uint8')
            for i in range(self.padded_act_size, self.num_actions):
                action_mask[i] = 1.
            return {
                'action_mask': action_mask,
                'observation': np.zeros(self.obs_size, dtype='float32'),
            }
        else:
            old_obss = self.sub_env.observe(agent)
            new_act_mask = np.zeros(self.num_actions, dtype='uint8')
            new_obs = np.zeros(self.obs_size, dtype='float32')
            act_mask = old_obss['action_mask']
            obs = old_obss['observation']
            new_act_mask[:len(act_mask)] = act_mask
            new_obs[:len(obs)] = obs
            new_obs[self.padded_obs_size:self.padded_obs_size+self.game_indicator_size] = 1.
            new_obs[self.padded_obs_size+self.game_indicator_size:self.obs_size] = 1.

            return {
                'action_mask': new_act_mask,
                'observation': new_obs,
            }

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        for env in self.sub_envs:
            env.seed(seed)

    def render(self, mode='human'):
        if self.sub_env is None:
            print("No env selected yet")
        else:
            return self.sub_env.render(mode)

    def close(self):
        pass







def perf_go_comparison():
    from pettingzoo.test import performance_benchmark
    from pettingzoo.classic import connect_four_v3
    spiel_env = OpenSpielEnv("connect_four")
    zoo_env = connect_four_v3.env()

    print("Speil env:")
    performance_benchmark(spiel_env)
    print("PettingZoo env:")
    performance_benchmark(zoo_env)

# perf_go_comparison()

def test_openspiel_env():
    from pettingzoo.test import api_test
    import random
    env = OpenSpielEnv("go", {"board_size":7,'komi':5.5})
    api_test(env, num_cycles=1000)
    env = OpenSpielEnv("connect_four")
    api_test(env, num_cycles=1000)
    env = OpenSpielEnv("breakthrough")
    api_test(env, num_cycles=1000)
    print("passed!")

def test_change_game_env():
    from pettingzoo.test import api_test
    env1 = OpenSpielEnv("go", {"board_size":7,'komi':5.5})
    env2 = OpenSpielEnv("connect_four")
    env3 = OpenSpielEnv("breakthrough")
    envs = [env1, env2, env3]
    env = ChangeGameEnv(envs)
    api_test(env, num_cycles=10000)

# test_change_game_env()

def test_openspiel_fns():
    games_list = pyspiel.registered_games()
    print(games_list)
    game = pyspiel.load_game("go", {"board_size":7})
    state = game.new_initial_state()
    print(state.observation_tensor(0))
    state.apply_action(0)
    print(state.current_player())
    print(state.is_chance_node())
    print(state.legal_actions_mask())
    # print(game.num_distinct_actions())
    # print(game.policy_tensor_shape())
    # print(state.rewards())
    # print(game.observation_tensor_size())
# test_openspiel_fns()
