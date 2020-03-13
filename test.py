from envs.surround.surround import Surround
import random
import pygame

pygame.init()

#screen = pygame.display.set_mode([40*30,20*30])

env = Surround()
for i in range(1000):
    for p in range(2):
        a = random.randrange(5)
        env.take_action(a,p)

    env.step_env()
    if env.game_ended:
        env.reset()
    (env.observe(0))
    #cur_surface = env.render()

    #screen.blit(cur_surface, (0,0))

    #pygame.display.flip()
