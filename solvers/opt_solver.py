from .agents import BaseAgent
from .populations import HomogenousPopulation
import numpy as np

class BasicGamesAgent(BaseAgent):
    def __init__(self,my_choice):
        self.my_choice = my_choice

    def step(self,obs):
        return self.my_choice


class OptRepPopulation(HomogenousPopulation):
    def __init__(self,my_objective,my_choicemixture):
        self.my_objective = my_objective
        self.my_choicemixture = my_choicemixture
        init_strat = my_objective.random_response()
        for p in range(2):
            self.my_choicemixture.add_player_choice(p,init_strat)

    def evaluate_sample(self):
        return BasicGamesAgent(self.my_choicemixture.sample(0))

    def optimal_response(self):
        my_evals = []
        my_strats = self.my_objective.all_opt_choices()
        for my_strat in my_strats:
            cur_player = 0
            my_evals.append(self.my_choicemixture.value_of(cur_player,my_strat))
        best_idx = np.argmax(np.array(my_evals))
        return my_strats[best_idx]

    def train_sample(self):
        choice1 = self.my_objective.random_response()#self.my_choicemixture.sample_p1(1)[0]
        player_resp = 1
        choice2 = self.optimal_response()#self.my_choicemixture.sample_player_opponents(player=1,sample_size=1)[0]
        return [BasicGamesAgent(choice1),BasicGamesAgent(choice2)],None

    def addExperiences(self,info,agents,result,observations,actions):
        best_agent = 1
        for p in range(2):
            self.my_choicemixture.add_player_choice(p,agents[best_agent].my_choice)
