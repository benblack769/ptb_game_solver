import numpy as np
import random
import copy
from .nash_finder import p1_solution,p2_solution


def calc_nash(objective,player):
    sol_finder = p1_solution if player == 0 else p2_solution
    support = sol_finder(objective)
    return support


# class NashChoiceObjective:
#     def __init__(self,game_objective,player,opponenet_strats):
#         self.opponent_strategies = opponenet_strats
#         self.game_objective = game_objective
#         self.player = player
#
#     def evaluate_strat(self,strat):
#         pass

class NashMixture:
    def __init__(self,objective):
        self.player_strategies = [[],[]]
        self.objective_matrix = np.zeros((0,0))
        self.objective = objective
        self.nash_support = np.zeros(0)

    def add_player_choice(self,player,strategy):
        self.player_strategies[player].append(strategy)
        other_player = player^1

        new_objective_row = []
        for other_strat in self.player_strategies[other_player]:
            strats = [None]*2
            strats[player] = strategy
            strats[other_player] = other_strat
            evaluation = self.objective.evaluate(*strats)
            new_objective_row.append(evaluation)
        new_objective_row = np.expand_dims(np.array(new_objective_row),other_player)
        self.objective_matrix = np.concatenate([self.objective_matrix,new_objective_row],axis=other_player)
        if self.objective_matrix.size:
            self.nash_support = calc_nash(self.objective_matrix,player)

    def sample(self,player):
        strategy_support = self.nash_support[player]
        return random.choices(self.player_strategies[player],weights=strategy_support)[0]

    def value_of(self,player,strategy):
        strategy_support = self.nash_support[player]
        other_player_strats = self.player_strategies[1-player]
        total = 0
        for support,opp_strat in zip(strategy_support,other_player_strats):
            player_mulval = -(player * 2 - 1)
            total += player_mulval * self.objective.evaluate(strategy,opp_strat)
        return total


class RectifiedNashMixture(NashMixture):
    def __init__(self,objective):
        super().__init__(objective)

    def value_of(self,player,strategy):
        strategy_support = self.nash_support[player]
        other_player_strats = self.player_strategies[1-player]
        total = 0
        for support,opp_strat in zip(strategy_support,other_player_strats):
            player_mulval = -(player * 2 - 1)
            rect_eval = max(0,self.objective.evaluate(strategy,opp_strat))
            total += player_mulval * rect_eval
        return total

class WeaknessesMixture(NashMixture):
    def __init__(self,objective):
        super().__init__(objective)

    def value_of(self,player,strategy):
        strategy_support = self.nash_support[player]
        other_player_strats = self.player_strategies[1-player]
        total = 0
        for support,opp_strat in zip(strategy_support,other_player_strats):
            player_mulval = -(player * 2 - 1)
            rect_eval = min(0,self.objective.evaluate(strategy,opp_strat))
            total += player_mulval * rect_eval
        return total

class UniformMixture:
    def __init__(self,objective):
        self.player_strategies = [[],[]]
        self.objective = objective

    def add_player_choice(self,player,strategy):
        self.player_strategies[player].append(strategy)

    def sample(self,player):
        return random.choice(self.player_strategies[player])

    def value_of(self,player,strategy):
        other_player_strats = self.player_strategies[1-player]
        total = 0
        for opp_strat in other_player_strats:
            player_mulval = -(player * 2 - 1)
            total += player_mulval * self.objective.evaluate(strategy,opp_strat)
        return total / len(other_player_strats)
