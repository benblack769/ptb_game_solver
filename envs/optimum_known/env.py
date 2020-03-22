from ..basic_env import BasicEnv

class SingleStepEnv(BasicEnv):
    def __init__(self,objective):
        self.objective = objective
        self._game_over = False
        self.scorep1 = 0

    def reset(self):
        self._game_over = False
        self.scorep1 = 0

    def step_env(self,player_actions):
        self._game_over = True
        resultp1 = self.objective.evaluate(*player_actions)
        self.scorep1 = resultp1

    def game_over(self):
        return self._game_over

    def scores(self):
        return [self.scorep1,-self.scorep1]

    def render(self):
        pass
