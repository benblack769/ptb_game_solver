from envs.surround.surround import Surround
import random
import pygame

pygame.init()

#screen = pygame.display.set_mode([40*30,20*30])

env = Surround()
for i in range(1000):
    actions = [random.randrange(5) for p in range(2)]

    env.step_env(actions)
    if env.game_ended:
        env.reset()
    obs = env.observe(0)
    #cur_surface = env.render()

    #screen.blit(cur_surface, (0,0))

    #pygame.display.flip()
