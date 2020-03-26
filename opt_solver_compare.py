from envs.optimum_known.env import SingleStepEnv
from envs.optimum_known.basicgames import RPCCombObjective,RPCObjective,CombObjective

from solvers.choicemixtures import NashMixture,RectifiedNashMixture, \
                                WeaknessesMixture,UniformMixture
from solvers.opt_solver import OptRepPopulation
from solvers.local_pert_solver import SelfPlayPertPopulation,FictitiousPertPopulation,NashPertPopulation
from solvers.nash_finder import zero_sum_asymetric_nash
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

def evaluate_zero_sum_pops(env,pop1,pop2,NUM_SAMPS=10,EVALS=1,NUM_PLAYERS=2):
    pop1_samps = [pop1.evaluate_sample() for _ in range(NUM_SAMPS)]
    pop2_samps = [pop2.evaluate_sample() for _ in range(NUM_SAMPS)]
    eval_matrix = [[get_repeated_score(env,[pop1_samp,pop2_samp],NUM_ITERS=EVALS)[0]
                    for pop1_samp in pop1_samps]
                    for pop2_samp in pop2_samps]

    eval,pop1_support,pop2_support = zero_sum_asymetric_nash(eval_matrix)
    # print(np.array(eval_matrix))
    # print([str(samp.my_choice) for samp in pop1_samps])
    # print([str(samp.my_choice) for samp in pop2_samps])
    # print([str(pop2.evaluate_sample().my_choice) for _ in range(15)])
    # print(pop1_support)
    # print(pop2_support)
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
    env = SingleStepEnv(objective)
    pop2 = OptRepPopulation(objective,UniformMixture(objective))
    #pop2 = SelfPlayPertPopulation(objective)#objective,RectifiedNashMixture(objective))
    pop1 = NashPertPopulation(objective)#objective,RectifiedNashMixture(objective))
    num_iters = 100
    game_repeats = 1
    compare_iters = 300
    train_pop(env,pop1,num_iters*100,game_repeats)
    train_pop(env,pop2,num_iters,game_repeats)
    pop_result = evaluate_zero_sum_pops(env,pop1,pop2,NUM_SAMPS=4)
    print("pop1")
    print("\n".join([str(pop1.evaluate_sample().my_choice) for _ in range(15)]))
    print("pop2")
    print("\n".join([str(pop2.evaluate_sample().my_choice) for _ in range(15)]))
    print(pop_result)

def main():
    #objective_compare(RPCObjective())
    objective_compare(RPCCombObjective(10,5))

if __name__ == "__main__":
    main()
