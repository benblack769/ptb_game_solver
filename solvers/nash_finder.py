import scipy.optimize
import numpy as np

import math
import torch
import torch.nn.functional as F

def p1_solution(matrix):
    matrix = 0.1 +  matrix - np.min(matrix)
    sizey,sizex = matrix.shape
    c = np.ones(sizex)
    A_ub = -matrix
    b_ub = -np.ones(sizey)
    u = scipy.optimize.linprog(c,A_ub=A_ub,b_ub=b_ub,method="revised simplex").x
    probs = u / np.sum(u)
    return probs

def p2_solution(matrix):
    return p1_solution(-np.transpose(matrix))

def zero_sum_asymetric_nash(matrix):
    '''
    finds the nash for a matrix form two player game
    returns:
        - game evaluation
        - x dimention solution
        - y dimention solution
    '''
    matrix = np.asarray(matrix)
    p1_sol,p2_sol = p1_solution(matrix),p2_solution(matrix)
    game_eval = np.dot(p2_sol,np.dot(matrix,p1_sol))
    return game_eval,p1_sol,p2_sol

def softmax(x):
    x = x - np.max(x)
    expx = np.exp(x)
    return expx/np.sum(expx,axis=-1,keepdims=True)


# def convex_hull_enbiggener(matrix,reg_val=0.001):
#     matrix = np.asarray(matrix,dtype=np.float32)
#     # add noise for numerical stability
#     matrix = matrix + np.random.uniform(low=0.,high=1e-2,size=matrix.shape)
#     from scipy.spatial import ConvexHull
#     from scipy.optimize import minimize
#     ysize,xsize = matrix.shape
#     assert ysize == xsize
#     size = xsize
#     def fun(mat):
#         mat = mat.reshape([size,size])
#         O = softmax(mat)
#         A = matrix
#         points = O * A
#         #print(O)
#         bad_point = -1e1*np.ones([1,size])
#         simple_points = np.random.uniform(low=0.,high=1e-5,size=points.shape)
#         all_points = np.concatenate([points,bad_point,simple_points],axis=0)
#         #print(all_points)
#         #print(O)
#         hull = ConvexHull(all_points)
#         #print(hull.volume)
#         reg_cost = reg_val * np.sum(np.square(mat))
#         #print(hull.volume)
#         return -hull.volume + reg_cost
#     start = np.random.uniform(low=0.,high=1e-2,size=matrix.shape)
#     res = minimize(fun,start,method="Nelder-Mead",options={"fatol":0.000001})
#     return softmax(res.x.reshape([size,size]))

def graph_opt_finder(matrix,reg_val=0.001):
    '''
    finds the graph optimal value for a two player game

    by graph optimal, I mean that the diversity objective can mean that every strategy
    (a node in the graph) is creating a set of edge weights that add to 1, that represent
    the probability of choosing that player as its opponent.

    Call this edge set the desired opponent matrix, $$ O $$

    Each player (node) is rewared by its edge assignment as follows. Each node has
    a weight assignment that reflects how many edges point towards it. This assignment is
    exactly

    $$ O * (1/n) $$

    Recall that there is also an evaluation matrix $$ A $$

    The final reward is:

    $$ -(A elprod O) * inv(O * (1/n)) $$

    returns:
        - game evaluation
        - x dimention solution
        - y dimention solution
    '''
    matrix = np.asarray(matrix,dtype=np.float32)
    # add noise for numerical stability
    matrix += np.random.uniform(low=0.,high=1e-5,size=matrix.shape)

    ysize,xsize = matrix.shape
    assert ysize == xsize, "matrix needs to be square"
    size = xsize
    rect_matrix = np.maximum(0,matrix)
    init_matrix = np.random.normal(size=[size,size]).astype(np.float32)
    #strat_weights = torch.tensor(np.ones(size,dtype=np.float32),requires_grad=True)
    logit_matrix = torch.tensor(init_matrix,requires_grad=True)
    value_matrix = torch.tensor(rect_matrix, requires_grad=False)
    optimizer = torch.optim.LBFGS([logit_matrix])#,max_iter=100,tolerance_change=1e-6)

    #for x in range(10):
    def closure():
        optimizer.zero_grad()
        O = F.softmax(logit_matrix,dim=1)
        A = value_matrix
        #node_weights_in = ((torch.matmul(O, torch.ones([size]))))
        node_weights_out = ((torch.matmul(torch.ones([size]), O)))
        print("start")
        strat_weights = torch.ones(size)/size
        UPDATE_SIZE = 0.1
        for _ in range(20):
            add_strat_weights = torch.matmul(A * O,strat_weights)
            #sub_strat_weights = torch.matmul(strat_weights,A * O)
            update_vec = (add_strat_weights)#-sub_strat_weights)
            strat_weights = (1-UPDATE_SIZE)*strat_weights + UPDATE_SIZE*update_vec
            print(strat_weights)
            print("sums")
            print(np.sum(add_strat_weights.detach().numpy()))
            #print(np.sum(sub_strat_weights.detach().numpy()))
            #assert math.isclose(np.sum(add_strat_weights.detach().numpy()), np.sum(sub_strat_weights.detach().numpy()),rel_tol=1e-5)
            strat_weights /= torch.sum(strat_weights)
        print(strat_weights)
        in_rewards = torch.sum((A * O))
        out_costs = 0*torch.sum((strat_weights-node_weights_out)**2)
        reg_cost = reg_val * torch.sum(logit_matrix**2)

        loss = reg_cost - in_rewards + out_costs
        #print(loss)
        loss.backward()
        return loss
    for i in range(15):
        optimizer.step(closure)

    res = F.softmax(logit_matrix,dim=1).detach().numpy()
    return res


if __name__ == "__main__":
    test_matrix1 = np.array([
        [0,1,-1],
        [-1,0,1],
        [1,-1,0],
    ],dtype=np.float32)
    test_matrix2 = np.array([
        [0,0.5,-1,1],
        [-0.5,0,1,0.5],
        [1,-1,0,-1],
        [2,-3,0.1,0.5],
    ],dtype=np.float32)
    test_matrix3 = np.array([
        [0,0.1,-1],
        [-0.1,0,1],
        [1,-1,0],
    ],dtype=np.float32)
    test_matrix4 = np.array([
        [0,1],
        [1,0],
    ],dtype=np.float32)
    test_matrix4 = np.array([
        [0,1,-1,-0.6],
        [-1,0,1,-0.6],
        [1,-1,0,-0.6],
        [0.6,0.6,0.6,0]
    ],dtype=np.float32)
    print(convex_hull_enbiggener(test_matrix1,reg_val=0.1))
    print(convex_hull_enbiggener(test_matrix3,reg_val=0.1))
    print(convex_hull_enbiggener(test_matrix2,reg_val=0.1))
    print(convex_hull_enbiggener(test_matrix4,reg_val=0.001))
