from .agents import BaseAgent
from .populations import HomogenousPopulation
import numpy as np
import random

class BasicGamesAgent(BaseAgent):
    def __init__(self,my_choice):
        self.my_choice = my_choice

    def step(self,obs):
        return self.my_choice

class SelfPlayPertPopulation(HomogenousPopulation):
    def __init__(self,my_objective,NUM_PERTS=10,NUM_EVALS=10):
        self.main_agents = [my_objective.random_response()]
        self.NUM_PERTS = NUM_PERTS
        self.NUM_EVALS = NUM_EVALS
        self.init_perts()

    def main_agent(self):
        return self.main_agents[-1]

    def init_perts(self):
        self.pert_agents = [self.main_agent().random_alt() for _ in range(self.NUM_PERTS)]
        self.pert_uneval_counts = np.zeros(self.NUM_PERTS,dtype=np.int32)
        self.pert_evaled_counts = np.zeros(self.NUM_PERTS,dtype=np.int32)
        self.pert_evals = np.zeros(self.NUM_PERTS)

    def evaluate_sample(self):
        return BasicGamesAgent(self.main_agent())

    def next_pert_train(self):
        pert_counts = self.pert_uneval_counts + self.pert_evaled_counts
        return (None if np.all(np.greater(self.pert_evaled_counts,self.NUM_EVALS))
                        else int(np.argmin(pert_counts)))

    def train_sample(self):
        next_pert_idx = self.next_pert_train()
        # if need to set main to pert, do so and get next pert
        if next_pert_idx is None:
            self.main_agents.append(self.pert_agents[np.argmax(self.pert_evals)])
            self.init_perts()
            next_pert_idx = self.next_pert_train()
            #print("agents incremented")
            #print(" ".join([str(agent) for agent in self.main_agents]))

        self.pert_uneval_counts[next_pert_idx] += 1
        return [BasicGamesAgent(self.main_agent()),BasicGamesAgent(self.pert_agents[next_pert_idx])],next_pert_idx

    def addExperiences(self,info,agents,result,observations,actions):
        self.pert_evals[info] += -result
        self.pert_uneval_counts[info] -= 1
        self.pert_evaled_counts[info] += 1


class FictitiousPertPopulation(SelfPlayPertPopulation):
    def main_agent(self):
        return random.choice(self.main_agents)
