import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from envs.optimum_known.env import SingleStepEnv
from envs.optimum_known.basicgames import RPCCombObjective,RPCObjective,CombObjective,BlottoCombObjective,BlottoObjective

from solvers.choicemixtures import NashMixture,RectifiedNashMixture, \
                                WeaknessesMixture,UniformMixture
from solvers.opt_solver import OptRepPopulation
from solvers.local_pert_solver import SelfPlayPertPopulation,FictitiousPertPopulation,NashPertPopulation,RectifiedNashPertPop,SoftPertPop
from solvers.nash_finder import zero_sum_asymetric_nash
import numpy as np
import multiprocessing
import random
import pandas as pd
import torch
torch.set_num_threads(1)

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
    starter = objective.random_response()
    pop2 = OptRepPopulation(objective,UniformMixture(objective))
    #pop2 = SelfPlayPertPopulation(objective)#objective,RectifiedNashMixture(objective))
    pop1 = SoftPertPop(starter,REG_VAL=0.03,NUM_PERTS=10,NUM_EVALS=1,POP_SIZE=25)#objective,RectifiedNashMixture(objective))
    #pop2 = RectifiedNashPertPop(starter,NUM_PERTS=10,NUM_EVALS=1,POP_SIZE=10)#objective,RectifiedNashMixture(objective))
    #pop2 = NashPertPopulation(objective,NUM_PERTS=10,NUM_EVALS=1,POP_SIZE=25)#objective,RectifiedNashMixture(objective))
    num_iters = 100
    game_repeats = 1
    compare_iters = 300
    train_pop(env,pop1,num_iters*1000,game_repeats)
    #p#rint("trained pop1")
    train_pop(env,pop2,num_iters*1,game_repeats)
    pop_result = evaluate_zero_sum_pops(env,pop1,pop2,NUM_SAMPS=300)

    #print([obj.match_choice for obj in objective.comb_objectives])
    print("pop1")
    print("\n".join([f"{pop1.current_pop[i]},  {pop1.nash_support[i]}" for i in range(10)]))
    print("pop2")
    #print()
    #print("\n".join([f"{pop2.current_pop[i]},  {pop2.nash_support[i]}" for i in range(10)]))
    #print("\n".join([str(pop2.evaluate_sample().my_choice) for _ in range(10)]))
    return (pop_result)

def evaluate_pop_vs(objective,compare_pop,arg_pop,train_iters):
    np.random.seed(random.randrange(1<<31))

    env = SingleStepEnv(objective)
    game_repeats = 1
    num_samps = 50
    train_pop(env,arg_pop,train_iters,game_repeats)
    pop_result = evaluate_zero_sum_pops(env,arg_pop,compare_pop,NUM_SAMPS=num_samps)
    return pop_result

def evaluate_pop_vs_comp(tuple):
    torch.set_num_threads(1)
    return evaluate_pop_vs(*tuple)

def generate_csv(objective, plot_pops, num_restarts, compare_pop, fname, MAX_ITERS=150000):
    env = SingleStepEnv(objective)
    evals = []
    pop_name = []
    num_eval_name = []
    iters = 1000
    all_proc_vals = []
    iter_buckets = 0
    while iters < MAX_ITERS:
        for restart in range(num_restarts):
            sample = objective.random_response()
            for pop_fact in plot_pops:
                pop = pop_fact(sample)
                all_proc_vals.append((objective, compare_pop, pop, iters))
        iters *= 2
        iter_buckets += 1

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(evaluate_pop_vs_comp,all_proc_vals)
    results = np.array(results).reshape((iter_buckets,num_restarts,len(plot_pops)))
    avg_results = np.mean(results,axis=1)
    std_results = np.std(results,axis=1)/np.sqrt(num_restarts)
    pop_avg_result = np.transpose(avg_results)
    pop_std_result = np.transpose(std_results)
    sample = objective.random_response()
    pop_names = [pop_fact(sample).__class__.__name__ for pop_fact in plot_pops]
    std_names = [pop_name + "_std" for pop_name in pop_names]
    all_names = pop_names + std_names
    all_results = np.concatenate([pop_avg_result,pop_std_result],axis=0)
    print(all_names)
    print(all_results.shape)
    data = {name:d  for name,d in zip(all_names,all_results)}
    df = pd.DataFrame(data=data)
    df.to_csv(fname,index=False)

def generate_csvs():
    objectives = [
        BlottoCombObjective(7,10),
        RPCCombObjective(10,5),
        CombObjective(25),
        BlottoObjective(10),
    ]
    for objective in objectives:
        compare_pop = OptRepPopulation(objective,UniformMixture(objective))

        env = SingleStepEnv(objective)
        compare_train_iters = 20
        game_repeats = 1
        train_pop(env,compare_pop,compare_train_iters,game_repeats)

        NUM_RESTARTS = 20
        NUM_EVALS = 30
        for pop_size in [10, 25]:
            populations = [
                lambda starter: SoftPertPop(starter,REG_VAL=0.03,NUM_PERTS=10,NUM_EVALS=NUM_EVALS,POP_SIZE=pop_size),
                lambda starter: RectifiedNashPertPop(starter,NUM_PERTS=10,NUM_EVALS=NUM_EVALS,POP_SIZE=pop_size),
                lambda starter: NashPertPopulation(starter,NUM_PERTS=10,NUM_EVALS=NUM_EVALS,POP_SIZE=pop_size),
                lambda starter: SelfPlayPertPopulation(starter),
                lambda starter: FictitiousPertPopulation(starter),
            ]
            NUM_POPS = 5
            csv_name = "outputs/{}_popsize{}.csv".format(str(objective),pop_size)
            generate_csv(objective, populations, NUM_RESTARTS, compare_pop, csv_name)

def objective_multi_compare(objective,num_reruns):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    results = pool.map(objective_compare,[objective for _ in range(num_reruns)])
    return np.mean(results),np.std(results)/np.sqrt(num_reruns)

def main_test():
    #objective_compare(RPCObjective())
    #objective_compare(RPCCombObjective(10,5))
    #print(objective_compare(BlottoCombObjective(7,10),24))
    #obj = RPCCombObjective(10,5)

    obj = BlottoCombObjective(7,10)
    #obj = CombObjective(25)
    #print(objective_compare(obj))
    print(objective_multi_compare(obj,24*2))
    #print(obj.match_choice)

if __name__ == "__main__":
    #main_test()
    generate_csvs()
