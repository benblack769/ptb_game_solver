import numpy as np
import random
import copy

class RPCChoice:
    def __init__(self,intval):
        self.feature = intval
    def random_choice():
        return RPCChoice(random.randrange(3))
    def random_alt(self):
        return RPCChoice(random.randrange(3))
    def __repr__(self):
        return str(self.feature)
    __str__ = __repr__

def normalize(vec):
    return vec / np.sqrt(np.sum(np.square(vec)))

class DiskChoice:
    def __init__(self,featurevec):
        self.feature = featurevec
    def random_choice():
        rad = random.random()
        theta = random.random()*2*math.pi
        x = math.cos(theta)*rad
        y = math.sin(theta)*rad
        return DiskChoice(np.array([x,y]))
    def random_alt(self):
        ALT_DIST = 0.1
        new_feature = self.feature + np.random.uniform(size=2)*ALT_DIST
        mag = np.sum(np.square(new_feature))
        if mag > 1:
            new_feature /= mag
        return DiskChoice(new_feature)
    def __repr__(self):
        return str(self.feature)
    __str__ = __repr__

class BlottoChoice:
    def __init__(self,alloc_choices):
        self.feature = alloc_choices
    def random_choice(game_size):
        return normalize(np.random.normal(size=game_size))

class CombChoice:
    def __init__(self,featurevec):
        self.num_choices = len(featurevec)
        self.feature = featurevec
    def random_choice(num_choices):
        return CombChoice(np.random.randint(0,1+1,size=num_choices))
    def random_alt(self):
        idx_flip = np.random.randint(0,self.num_choices)
        new_feature = np.copy(self.feature)
        new_feature[idx_flip] = 1-new_feature[idx_flip]
        return CombChoice(new_feature)
    def __repr__(self):
        return str(self.feature)
    __str__ = __repr__

class RPC_CombChoice:
    def __init__(self,comb,rpc):
        self.num_choices = comb.num_choices
        self.comb = comb
        self.rpc = rpc
    def random_choice(num_choices):
        return RPC_CombChoice(CombChoice.random_choice(num_choices),RPCChoice.random_choice())
    def random_alt(self):
        if random.random() < 1./(self.num_choices+1):
            return RPC_CombChoice(self.comb,self.rpc.random_alt())
        else:
            return RPC_CombChoice(self.comb.random_alt(),self.rpc)
    def __repr__(self):
        return "rpc: {}, comb: {}".format(self.rpc,self.comb)
    __str__ = __repr__

class CombObjective:
    def __init__(self,num_combs):
        self.num_combs = num_combs
        self.match_choice = CombChoice.random_choice(num_combs)
    def sing_eval(self,choice):
        return np.sum(np.equal(self.match_choice.feature,choice.feature).astype(np.int32))
    def evaluate(self,choice1,choice2):
        return self.sing_eval(choice1) - self.sing_eval(choice2)
    def all_opt_choices(self):
        return self.match_choice
    def random_response(self):
        return CombChoice.random_choice(self.num_combs)

class RPCObjective:
    def __init__(self):
        pass
    def evaluate(self,choicep1,choicep2):
        diffmod3 = (choicep1.feature - choicep2.feature)%3
        return diffmod3 if diffmod3 <= 1 else -1
    def all_opt_choices(self):
        return [RPCChoice(c) for c in range(3)]
    def random_response(self):
        return RPCChoice.random_choice()

class RPCCombObjective:
    def __init__(self,num_combs,mul_val):
        self.num_combs = num_combs
        self.comb_objectives = [CombObjective(num_combs) for _ in range(3)]
        self.rpc = RPCObjective()
        self.mul_val = mul_val

    def evaluate(self,choicep1,choicep2):
        valp1 = self.comb_objectives[choicep1.rpc.feature].sing_eval(choicep1.comb)
        valp2 = self.comb_objectives[choicep2.rpc.feature].sing_eval(choicep2.comb)
        rpc_val = self.rpc.evaluate(choicep1.rpc,choicep2.rpc)
        return valp1 - valp2 + rpc_val * self.mul_val

    def all_opt_choices(self):
        return [RPC_CombChoice(self.comb_objectives[i].match_choice,RPCChoice(i)) for i in range(3)]

    def random_response(self):
        return RPC_CombChoice.random_choice(self.num_combs)
