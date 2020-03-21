'''
A Partially observable version of the old atari surround game
'''
import numpy as np
from ..basic_env import BasicEnv


EMPTY = 0
FULL = 1

def action_to_dir(action,olddir):
    if action == 0:
        return olddir
    else:
        if action == 1:
            return (-1,0)
        if action == 2:
            return (1,0)
        if action == 3:
            return (0,1)
        if action == 4:
            return (0,-1)
    assert False,"bad action to Surround"

def add(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return (x1+x2,y1+y2)


class Surround(BasicEnv):
    def __init__(self,BOARD_SIZE=(40,20),VIEW_SIZE=5):
        self.board_size = BOARD_SIZE
        self.view_size = VIEW_SIZE
        self.board = []

        BSX,BSY = self.board_size
        self.posses = []
        self.dirs = []

        self.game_ended = False
        self.winner = None

        self.reset()

    def reset(self):
        self.game_ended = False
        self.winner = None

        BSX,BSY = self.board_size
        self.board = [[EMPTY for _ in range(BSX)] for _ in range(BSY)]

        self.posses = [(BSX//3,BSY//2), ((BSX*2)//3,BSY//2)]
        self.dirs = [(1,0), (-1,0)]

        # fill in border
        for y in [0,BSY-1]:
            for x in range(0,BSX):
                self.board[y][x] = FULL
        for x in [0,BSX-1]:
            for y in range(0,BSY):
                self.board[y][x] = FULL

    def step_env(self,player_actions):
        assert len(player_actions) == 2
        for player, action in enumerate(player_actions):
            self.dirs[player] = new_dir = action_to_dir(action,self.dirs[player])

        for p in range(2):
            ox,oy = self.posses[p]
            new_pos = add(self.posses[p],self.dirs[p])
            nx,ny = new_pos
            if self.board[ny][nx] == FULL:
                self.game_ended = True
                self.winner = p^1
                break

            self.board[oy][ox] = FULL
            self.posses[p] = new_pos

    def observe(self,player):
        BSX,BSY = self.board_size

        v = self.view_size
        px,py = self.posses[player]
        pan1 = np.zeros((2*v+1,2*v+1))
        for y in range(max(0,py-v),min(BSY,py+v+1)):
            for x in range(max(0,px-v),min(BSX,px+v+1)):
                if self.board[y][x]:
                    pan1[y-py+v,x-px+v] = 1

        pos_panel_cur = np.zeros_like(pan1)
        pos_panel_cur[v,v] = 1

        pos_panel_other = np.zeros_like(pan1)
        ox,oy = self.posses[player^1]
        offx,offy = (ox-px+v,oy-py+v)
        if 0 <= offx <= 2*v and 0 <= offy <= 2*v:
            pos_panel_other[offx,offy] = 1

        all_pannels = np.stack([pan1,pos_panel_cur,pos_panel_other],axis=0)
        return all_pannels

    def render(self):
        import pygame
        BSX,BSY = self.board_size
        BOX_PIX = BP = 30

        #pygame.init()
        screen = pygame.Surface((BSX*BOX_PIX, BSY*BOX_PIX), pygame.SRCALPHA)

        screen.fill((138, 115, 55))
        for y in range(BSY):
            for x in range(BSX):
                if self.board[y][x]:
                    screen.fill((232, 196, 97),rect=(x*BP,y*BP,BP,BP))

        hidden_screen = pygame.Surface(screen.get_size(), pygame.SRCALPHA)  # the size of your rect
        #hidden_screen.set_alpha(32)
        hidden_screen.fill((0, 0, 0, 32))

        v = self.view_size
        bx,by = self.board_size
        colors = [(0,0,255),(0,255,0)]
        for (px,py),color in zip(self.posses,colors):
            for y in range(max(0,py-v),min(by,py+v+1)):
                for x in range(max(0,px-v),min(bx,px+v+1)):
                    hidden_screen.fill(color+(32,), (x*BP,y*BP,BP,BP))
            hidden_screen.fill(color, (px*BP,py*BP,BP,BP))

        screen.blit(hidden_screen, (0,0))

        return screen
