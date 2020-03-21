from envs.surround.surround import Surround
import random
import pygame

pygame.init()

#screen = pygame.display.set_mode([40*30,20*30])

env = Surround()
total_score = 0
num_games = 0
for i in range(100):
    actions = [random.randrange(5) for p in range(2)]

    env.step_env(actions)
    if env.game_over():
        total_score += env.scores()[0]
        num_games += 1
        env.reset()
        print(total_score/num_games)

    obs = env.observe(0)
    #cur_surface = env.render()

    #screen.blit(cur_surface, (0,0))

    #pygame.display.flip()
