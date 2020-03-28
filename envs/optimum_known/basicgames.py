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
    assert np.all(np.greater_equal(vec,0))
    return vec / np.sum(vec)

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
        return BlottoChoice(normalize(np.random.exponential(size=game_size)))
    def random_alt(self):
        new_feature = np.copy(self.feature)
        SUB_VAL = 1./(4*len(new_feature))
        sub_idx = random.randrange(len(new_feature))
        add_idx = random.randrange(len(new_feature))
        transfer_val = min(new_feature[sub_idx],SUB_VAL)
        new_feature[sub_idx] -= transfer_val
        new_feature[add_idx] += transfer_val
        assert  0.9999 < np.sum(new_feature) < 1.0001
        return BlottoChoice(new_feature)
    def __repr__(self):
        return str(self.feature)
    __str__ = __repr__

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

class Blotto_CombChoice:
    def __init__(self,comb,blotto):
        self.num_choices = comb.num_choices
        self.game_size = len(blotto.feature)
        self.comb = comb
        self.blotto = blotto
    def random_choice(num_choices,game_size):
        return Blotto_CombChoice(CombChoice.random_choice(num_choices),BlottoChoice.random_choice(game_size))
    def random_alt(self):
        if random.random() < 0.5:
            return Blotto_CombChoice(self.comb,self.blotto.random_alt())
        else:
            return Blotto_CombChoice(self.comb.random_alt(),self.blotto)
    def __repr__(self):
        return "blotto: {}, comb: {}".format(self.blotto,self.comb)
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

class BlottoObjective:
    def __init__(self,game_size):
        self.game_size = game_size
    def evaluate(self,choicep1,choicep2):
        assert self.game_size == len(choicep1.feature)
        p1_score = np.sum(np.greater(choicep1.feature,choicep2.feature).astype(np.int32))
        p2_score = np.sum(np.less(choicep1.feature,choicep2.feature).astype(np.int32))
        anti_symetric_score = p1_score - p2_score
        return anti_symetric_score
    def all_opt_choices(self):
        return [BlottoChoice.random_choice() for c in range(300)]
    def random_response(self):
        return BlottoObjective.random_choice()

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

class BlottoCombObjective:
    def __init__(self,num_combs,game_size):
        self.game_size = game_size
        self.num_combs = num_combs
        self.comb_objectives = [CombObjective(num_combs) for _ in range(game_size)]
        self.blotto = BlottoObjective(game_size)

    def evaluate(self,choicep1,choicep2):
        comb_scores_p1 = np.array([obj.sing_eval(choicep1.comb) for obj in self.comb_objectives])
        comb_scores_p2 = np.array([obj.sing_eval(choicep2.comb) for obj in self.comb_objectives])
        p1_score = (np.greater(choicep1.blotto.feature,choicep2.blotto.feature).astype(np.int32))
        p2_score = (np.less(choicep1.blotto.feature,choicep2.blotto.feature).astype(np.int32))
        blotto_score = p1_score*comb_scores_p1 - p2_score*comb_scores_p2
        return np.sum(blotto_score)

    def all_opt_choices(self):
        return [Blotto_CombChoice(CombChoice.random_choice(self.num_combs),BlottoChoice.random_choice(self.game_size)) for _ in range(100)]

    def random_response(self):
        return Blotto_CombChoice.random_choice(self.num_combs,self.game_size)
