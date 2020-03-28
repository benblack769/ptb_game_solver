from envs.optimum_known.env import SingleStepEnv
from envs.optimum_known.basicgames import RPCCombObjective,RPCObjective,CombObjective,BlottoCombObjective

from solvers.choicemixtures import NashMixture,RectifiedNashMixture, \
                                WeaknessesMixture,UniformMixture
from solvers.opt_solver import OptRepPopulation
from solvers.local_pert_solver import SelfPlayPertPopulation,FictitiousPertPopulation,NashPertPopulation,RectifiedNashPertPop
from solvers.nash_finder import zero_sum_asymetric_nash
import numpy as np
import multiprocessing
import random

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

def evaluate_zero_sum_pops(env,pop1,pop2,NUM_SAMPS=10,EVALS=1,NUM_PLAYERS=2):
    pop1_samps = [pop1.evaluate_sample() for _ in range(NUM_SAMPS)]
    pop2_samps = [pop2.evaluate_sample() for _ in range(NUM_SAMPS)]
    eval_matrix = [[get_repeated_score(env,[pop1_samp,pop2_samp],NUM_ITERS=EVALS)[0]
                    for pop1_samp in pop1_samps]
                    for pop2_samp in pop2_samps]

    eval,pop1_support,pop2_support = zero_sum_asymetric_nash(eval_matrix)
    return eval

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
    # random seed set for correct multiprocessing
    np.random.seed(random.randrange(1<<31))

    env = SingleStepEnv(objective)
    pop2 = OptRepPopulation(objective,UniformMixture(objective))
    #pop2 = SelfPlayPertPopulation(objective)#objective,RectifiedNashMixture(objective))
    pop1 = RectifiedNashPertPop(objective)#objective,RectifiedNashMixture(objective))
    #pop2 = NashPertPopulation(objective)#objective,RectifiedNashMixture(objective))
    num_iters = 30
    game_repeats = 1
    compare_iters = 300
    train_pop(env,pop1,num_iters*100,game_repeats)
    #p#rint("trained pop1")
    train_pop(env,pop2,num_iters,game_repeats)
    pop_result = evaluate_zero_sum_pops(env,pop1,pop2,NUM_SAMPS=4)
    return pop_result
    #print("pop1")
    #print("\n".join([str(pop1.evaluate_sample().my_choice) for _ in range(15)]))
    #print("pop2")
    #print("\n".join([str(pop2.evaluate_sample().my_choice) for _ in range(15)]))
    #print(pop_result)

def objective_multi_compare(objective,num_reruns):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    results = pool.map(objective_compare,[objective for _ in range(num_reruns)])
    return np.mean(results),np.std(results)/np.sqrt(num_reruns)

def main():
    #objective_compare(RPCObjective())
    #objective_compare(RPCCombObjective(10,5))
    print(objective_multi_compare(BlottoCombObjective(7,10),48))

if __name__ == "__main__":
    main()
