class BasicEnv:
    def __init__(self,*options,**kwoptions):
        '''
        sets up game with options
        '''

    def reset(self):
        '''
        resets game to initial state according to the options
        '''

    def step_env(self,player_actions):
        '''
        steps env with players having taken actions
        '''

    def observe(self,player):
        '''
        return observation for player
        '''

    def render(self):
        '''
        display game for players
        '''
