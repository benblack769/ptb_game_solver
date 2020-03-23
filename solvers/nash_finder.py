import scipy.optimize
import numpy as np
#
# def dominated_row_elim(matrix):
#     sizey,sizex = matrix.shape
#     rem_idxs = []
#     for i in range(sizey):
#         for j in range(sizey):
#             if np.all(np.less(matrix[i],matrix[j])):
#                 rem_idxs.append(i)
#                 break
#     rem_idxs = set(rem_idxs)
#     keep_idxs = set(range(sizey)) - rem_idxs
#     return bool(rem_idxs), matrix[list(keep_idxs)]
#
# def dominated_col_elim(matrix):
#     sizey,sizex = matrix.shape
#     rem_idxs = []
#     for i in range(sizex):
#         for j in range(sizex):
#             if np.all(np.less(matrix[:,i],matrix[:,j])):
#                 rem_idxs.append(i)
#                 break
#     rem_idxs = set(rem_idxs)
#     keep_idxs = set(range(sizex)) - rem_idxs
#     return bool(rem_idxs), matrix[:,list(keep_idxs)]
#
# def elim_dominated(matrix):
#     row_worked = col_worked = True
#     while row_worked or col_worked:
#         row_worked,matrix = dominated_row_elim(matrix)
#         col_worked,matrix = dominated_col_elim(matrix)
#     return matrix

def p1_solution(matrix):
    matrix = 0.1 + matrix + np.min(matrix)
    sizey,sizex = matrix.shape
    c = np.ones(sizex)
    A_ub = -matrix
    b_ub = -np.ones(sizey)
    u = scipy.optimize.linprog(c,A_ub=A_ub,b_ub=b_ub).x
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


if __name__ == "__main__":
    test_matrix1 = np.array([
        [-1,0,1],
        [1,-1,0],
        [0,1,-1],
    ])
    test_matrix2 = np.array([
        [0,0.5,-1],
        [-0.5,0,1],
        [1,-1,0],
    ])
    print(zero_sum_asymetric_nash(test_matrix1))
    print(zero_sum_asymetric_nash(test_matrix2))
