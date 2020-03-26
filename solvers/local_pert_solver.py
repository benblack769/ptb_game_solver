from .agents import BaseAgent
from .populations import HomogenousPopulation
import numpy as np
import random
from .nash_finder import p1_solution

class BasicGamesAgent(BaseAgent):
    '''
    agent wrapper for basic games Choices
    '''
    def __init__(self,my_choice):
        self.my_choice = my_choice

    def step(self,obs):
        return self.my_choice

    def __repr__(self):
        return "agent: {}".format(self.my_choice)

    __str__ = __repr__

class TaskEvalulator:
    def __init__(self,task,NUM_EVALS):
        self.task = task
        self.NUM_EVALS = NUM_EVALS
        self.reward = 0
        self.started_count = 0
        self.finished_count = 0

    def all_allocated(self):
        return self.started_count >= self.NUM_EVALS

    def is_finished(self):
        return self.finished_count >= self.NUM_EVALS

    def get_reward(self):
        assert self.is_finished()
        return self.reward / self.finished_count

    def inc_task(self):
        self.started_count += 1

    def place_eval(self,task,reward):
        assert task == self.task
        self.finished_count += 1
        self.reward += reward

class EvalAllocator:
    def __init__(self,NUM_EVALS):
        self.NUM_EVALS = NUM_EVALS
        self.task_list = []
        self.task_mapping = {}
        self.finish_task_index = 0
        self.started_task_index = 0

    def add_tasks(self,tasks):
        for task in tasks:
            self.task_mapping[task] = len(self.task_list)
            self.task_list.append(TaskEvalulator(task,self.NUM_EVALS))

    def all_finished(self):
        return len(self.task_list) <= self.finish_task_index

    def any_finished(self):
        return self.finish_task_index > 0

    def pop_finished(self):
        res = [(task.get_reward(),task.task) for task in self.task_list[:self.finish_task_index]]
        self.task_list = self.task_list[self.finish_task_index:]
        self.started_task_index -= self.finish_task_index
        self.finish_task_index = 0
        return res

    def place_eval(self,task,reward):
        assert task == self.task_list[self.finish_task_index].task
        self.task_list[self.finish_task_index].place_eval(task,reward)
        if self.task_list[self.finish_task_index].is_finished():
            self.finish_task_index += 1

    def next_task(self):
        idx = min(len(self.task_list)-1,self.started_task_index)
        task = self.task_list[self.started_task_index]
        task.inc_task()
        if task.all_allocated():
            self.started_task_index += 1
        return task.task


class PertLearner:
    def __init__(self,init_choice,NUM_PERTS,NUM_EVALS):
        self.main_choice = init_choice
        self.trained_stack = []
        self.NUM_PERTS = NUM_PERTS
        self.NUM_EVALS = NUM_EVALS
        self.eval_alloc = EvalAllocator(NUM_EVALS)
        self.init_perts()

    def init_perts(self):
        self.pert_agents = [self.main_choice.random_alt() for _ in range(self.NUM_PERTS)]
        tasks = [(agent) for agent in range(self.NUM_PERTS)]
        self.eval_alloc.add_tasks(tasks)

    def evaluate_sample(self):
        return BasicGamesAgent(self.main_choice)

    def pop_trained_stack(self):
        trained_stack = self.trained_stack
        self.trained_stack = []
        return trained_stack

    def train_sample(self):
        # if need to set main to pert, do so and get next pert
        if self.eval_alloc.all_finished():
            task_list = self.eval_alloc.pop_finished()
            agent_reward = np.zeros(len(self.pert_agents))
            for rew,task in task_list:
                agent_reward[task] = rew
            self.main_choice = self.pert_agents[np.argmax(agent_reward)]
            self.trained_stack.append(self.main_choice)
            self.init_perts()

        task = self.eval_alloc.next_task()

        return self.pert_agents[task],task

    def experience_train(self,info,reward):
        self.eval_alloc.place_eval(info,reward)


class SelfPlayPertPopulation(HomogenousPopulation):
    def __init__(self,my_objective,NUM_PERTS=10,NUM_EVALS=10):
        self.main_agents = [my_objective.random_response()]
        self.cur_learner = PertLearner(self.main_agent(),NUM_PERTS,NUM_EVALS)
        self.NUM_PERTS = NUM_PERTS
        self.NUM_EVALS = NUM_EVALS

    def main_agent(self):
        return self.main_agents[-1]

    def evaluate_sample(self):
        return BasicGamesAgent(self.main_agent())

    def train_sample(self):
        pert_sample,info = self.cur_learner.train_sample()
        self.main_agents += self.cur_learner.pop_trained_stack()
        return [BasicGamesAgent(self.main_agent()),BasicGamesAgent(pert_sample)],info

    def addExperiences(self,info,agents,result,observations,actions):
        learner_reward = -result
        self.cur_learner.experience_train(info,learner_reward)


class FictitiousPertPopulation(SelfPlayPertPopulation):
    def main_agent(self):
        return random.choice(self.main_agents)


class NashPertPopulation(HomogenousPopulation):
    def __init__(self,my_objective,NUM_PERTS=10,NUM_EVALS=10,POP_SIZE=10):
        starter = my_objective.random_response()
        self.current_pop = [starter.random_alt() for _ in range(POP_SIZE)]
        self.nash_support = np.ones(POP_SIZE)/POP_SIZE
        self.POP_SIZE = POP_SIZE
        self.NUM_EVALS = NUM_EVALS
        self.NUM_PERTS = NUM_PERTS
        self.eval_alloc = EvalAllocator(NUM_EVALS)
        self.eval_matrix = np.zeros([self.POP_SIZE,self.POP_SIZE])
        self.queue_matrix_evals()

    def queue_matrix_evals(self):
        tasks = [("matrix",(p1,p2))
                    for p1 in range(self.POP_SIZE)
                        for p2 in range(self.POP_SIZE)]

        self.eval_alloc.add_tasks(tasks)

    def queue_pop_evals(self):
        self.pop_alts = [[choice.random_alt() for _ in range(self.NUM_PERTS)] for choice in self.current_pop]
        tasks = [("learn",(p,pert)) for p in range(self.POP_SIZE) for pert in range(self.NUM_PERTS)]
        self.eval_alloc.add_tasks(tasks)

    def recalc_nash(self):
        self.nash_support = p1_solution(self.eval_matrix)

    def evaluate_sample(self):
        return BasicGamesAgent(random.choices(self.current_pop,weights=self.nash_support)[0])

    def handle_task_completion(self):
        if self.eval_alloc.all_finished():
            tasks = self.eval_alloc.pop_finished()
            _,(t0name,_) = tasks[0]
            if t0name == "matrix":
                self.eval_matrix = np.zeros([self.POP_SIZE,self.POP_SIZE])
                for rew,task in tasks:
                    name,data = task
                    assert name == t0name
                    p1,p2 = data
                    self.eval_matrix[p1][p2] = rew
                self.recalc_nash()
                self.queue_pop_evals()
            elif t0name == "learn":
                pop_values = [[0 for _ in range(self.NUM_PERTS)] for choice in self.current_pop]
                for rew,task in tasks:
                    name,data = task
                    assert name == t0name
                    p1,pert = data
                    pop_values[p1][pert] += rew
                for i in range(self.POP_SIZE):
                    self.current_pop[i] = self.pop_alts[i][np.argmax(pop_values[i])]
                self.queue_matrix_evals()
            else:
                assert False, t0name

    def train_sample(self):
        name,data = task = self.eval_alloc.next_task()
        if name == "matrix":
            p1,p2 = data
            return [BasicGamesAgent(self.current_pop[p1]),BasicGamesAgent(self.current_pop[p2])],task
        else:
            p1,pert = data
            return [BasicGamesAgent(self.pop_alts[p1][pert]),self.pop_alt_compare(p1)],task

    def pop_alt_compare(self,pop_alt_idx):
        return self.evaluate_sample()

    def addExperiences(self,info,agents,result,observations,actions):
        self.eval_alloc.place_eval(info,result)
        self.handle_task_completion()

class RectifiedNashPertPop(NashPertPopulation):
    def pop_alt_compare(self,pop_alt_idx):
        win_val = np.maximum(0,self.eval_matrix[pop_alt_idx])
        support_val = self.nash_support
        target_mag = win_val * support_val
        target_mag /= (np.sum(target_mag)+1e-10)
        compare_choice = random.choices(self.current_pop,weights=target_mag)[0]
        return BasicGamesAgent(compare_choice)
