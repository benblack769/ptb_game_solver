from envs.surround.surround import Surround
import random
import pygame
import time
import numpy as np

pygame.init()

# screen = pygame.display.set_mode([40*30,20*30])

env = Surround()
total_score = 0
num_games = 0
while num_games < 10000:
    rand_act = random.randrange(5)
    actions = [rand_act for p in range(2)]

    env.step_env(actions)
    if env.game_over():
        total_score += env.scores()[0]
        num_games += 1
        env.reset()
        #print(total_score/num_games)

    #obss = [for p in range(2)]
    # assert np.all(np.equal(env.observe(0),env.observe(1) ))
    # cur_surface = env.render()
    # time.sleep(0.1)
    #
    # screen.blit(cur_surface, (0,0))
    #
    # pygame.display.flip()
