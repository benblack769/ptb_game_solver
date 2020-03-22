
class BaseAgent:
    def step(self,observation):
        '''
        stateful choice to move forward through game

        parameters:
            - observation: obs
                observation of current state in environment
        returns:
            action
        side_effect:
            updates internal state based off observation
        '''

    def reset(self):
        '''
        side_effect:
            clears internal state to play new game from scratch
        '''

    def addExperiences(self,observations,actions,reward):
        '''
        train agent on all experiences for a game

        parameters:
            - observations: all observations seen in game
            - actions: all actions seen in game
            - reward: reward for whole game
        '''
