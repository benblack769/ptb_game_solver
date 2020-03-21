import load_env

class env:
    def __init__(self,json_fname):
        self.libvis, self.env_values, self.guard, self.agent, self.rewards = load_env.load_env(json_fname)


    def step(self,action):
        pass
