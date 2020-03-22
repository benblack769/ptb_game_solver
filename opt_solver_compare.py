from envs.optimum_known.env import SingleStepEnv
from envs.optimum_known.basicgames import RPCCombObjective,RPCObjective,CombObjective

from solvers.choicemixtures import NashMixture,RectifiedNashMixture, \
                                WeaknessesMixture,UniformMixture
from solvers.opt_solver import OptRepPopulation
import numpy as np

def get_single_game_score(env,agents,NUM_PLAYERS=2):
    env.reset()
    while not env.game_over():
        observations = [env.observe(p) for p in range(NUM_PLAYERS)]
        actions = [agent.step(obs) for agent,obs in zip(agents,observations)]
        env.step_env(actions)

    scores = env.scores()
    env.reset()
    return np.array(scores)

def get_repeated_score(env,agents,NUM_ITERS,NUM_PLAYERS=2):
    accum_scores = np.zeros(NUM_PLAYERS)
    for x in range(NUM_ITERS):
        scores = get_single_game_score(env,agents)
        accum_scores += scores

    avg_scores = accum_scores / NUM_ITERS
    return avg_scores

def train_pop(env,pop,NUM_ITERS,NUM_GAME_REPEATS,NUM_PLAYERS=2):
    for x in range(NUM_ITERS):
        agents,info = pop.train_sample()
        scores = get_repeated_score(env,agents,NUM_GAME_REPEATS)
        pop.addExperiences(info,agents,scores[0],None,None)
        print(pop.my_choicemixture.player_strategies)

def compare_populations(env,pop1,pop2,ITERS=1000,NUM_PLAYERS=2):
    pops = [pop1,pop2]
    accum_scores = np.zeros(NUM_PLAYERS)
    for x in range(ITERS):
        agents = [pop.evaluate_sample() for pop in pops]
        scores = get_single_game_score(env,agents)
        accum_scores += scores

    avg_scores = accum_scores/ITERS
    return avg_scores


def objective_compare(objective):
    env = SingleStepEnv(objective)
    pop2 = OptRepPopulation(objective,UniformMixture(objective))
    pop1 = OptRepPopulation(objective,RectifiedNashMixture(objective))
    num_iters = 7
    game_repeats = 1
    compare_iters = 300
    train_pop(env,pop1,num_iters,game_repeats)
    train_pop(env,pop2,num_iters,game_repeats)
    pop_result = compare_populations(env,pop1,pop2,compare_iters)
    print(pop_result)

def main():
    objective_compare(RPCObjective())

if __name__ == "__main__":
    main()
