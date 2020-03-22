
class PopAgent:
    '''
    the agents returned by train_sample() are of this type
    '''
    def __init__(self,idx,agent):
        self.agent = agent
        self.idx = idx

    def get_agent(self):
        return agent

class HomogenousPopulation:
    def evaluate_sample(self):
        '''
        returns:
            - agent:
                agent for cross population comparisons
        '''

    def addExperiences(self,info,agents,result,observations,actions):
        '''
        parameters:
            - info: population specific info from train_sample()
            - agents: list of agents
            - result: float
                result (homogenous, so single value applies to all agents)
            - observations: <list<list<obs>>
                all observations seen by the agents over the game
            - acions: <list<list<actions>>
                all actions taken by the agents
        '''
        pass

    def train_sample(self):
        '''
        returns:
            - [agent,agent]: list of two agents to play game vs each other
            - info: population specific data to be passed into addExperiences
        '''
        pass
