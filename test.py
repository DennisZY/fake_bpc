from utils import gen_tetrad_func, gauss_ci_test_generator
import numpy as np
import pandas as pd
from collections import deque
from itertools import combinations, permutations

if __name__ == '__main__':
    # np.random.seed(0)
    # l_size = 1
    # o_size = 4
    # sample_size = 200
    # v_labels = ['L' + str(i + 1) for i in range(l_size)] + ['X' + str(i + 1) for i in range(o_size)]
    # has_edge = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)
    # has_edge.loc['L1', 'X1'] = True
    # has_edge.loc['L1', 'X2'] = True
    # has_edge.loc['L1', 'X3'] = True
    # has_edge.loc['L1', 'X4'] = True
    # in_degree = [0] * (l_size + o_size)
    #
    # for i in v_labels:
    #     for ind, j in enumerate(v_labels):
    #         if has_edge.loc[i, j]:
    #             in_degree[ind] += 1
    # df = pd.DataFrame(np.concatenate((np.random.normal(0, 0.3 ** 2, size=(sample_size, l_size)),
    #                                   np.random.normal(0, 0.3 ** 2, size=(sample_size, o_size))), axis=1),
    #                   index=range(sample_size), columns=v_labels, dtype=float)
    # q = deque()
    # for ind, val in enumerate(in_degree):
    #     if val == 0:
    #         q.append(v_labels[ind])
    #
    # while len(q) != 0:
    #     top = q.popleft()
    #     for ind, i in enumerate(has_edge.loc[top, :]):
    #         if i:
    #             df.loc[:, v_labels[ind]] += np.random.uniform(0.2, 0.8) * df.loc[:, top]
    #             in_degree[ind] -= 1
    #             if in_degree[ind] == 0:
    #                 q.append(v_labels[ind])
    # df = df.loc[:, 'X1':]
    # tetrad_score1, tetrad_score3, tetrad_holds = gen_tetrad_func(df, 0.001)
    # ci_test = gauss_ci_test_generator(df,0.01)
    # for (i,j) in combinations(df.columns.to_list(),2):
    #     print(i,j,ci_test(i,j,[]))
    # print(tetrad_score3('X1', 'X2', 'X3', 'X4'))


    l_size = 2
    o_size = 4
    sample_size = 200
    v_labels = ['L' + str(i + 1) for i in range(l_size)] + ['X' + str(i + 1) for i in range(o_size)]
    has_edge = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)

    has_edge.loc['L1', 'L2'] = True
    has_edge.loc['L1', 'X1'] = True
    has_edge.loc['L1', 'X2'] = True
    has_edge.loc['L2', 'X3'] = True
    has_edge.loc['L2', 'X4'] = True
    has_edge.loc['X3', 'X4'] = True
    in_degree = [0] * (l_size + o_size)

    for i in v_labels:
        for ind, j in enumerate(v_labels):
            if has_edge.loc[i, j]:
                in_degree[ind] += 1
    df = pd.DataFrame(np.concatenate((np.random.normal(0, np.random.uniform(1,3), size=(sample_size, l_size)),
                                      np.random.normal(0, np.random.uniform(1,3), size=(sample_size, o_size))), axis=1),
                      index=range(sample_size), columns=v_labels, dtype=float)
    q = deque()
    for ind, val in enumerate(in_degree):
        if val == 0:
            q.append(v_labels[ind])

    while len(q) != 0:
        top = q.popleft()
        for ind, i in enumerate(has_edge.loc[top, :]):
            if i:
                df.loc[:, v_labels[ind]] += np.random.uniform(0.5, 1.5) * df.loc[:, top]
                in_degree[ind] -= 1
                if in_degree[ind] == 0:
                    q.append(v_labels[ind])
    df = df.loc[:, 'X1':]
    tetrad_score1, tetrad_score3, tetrad_holds = gen_tetrad_func(df, 0.01)
    ci_test = gauss_ci_test_generator(df,0.1)
    for (i,j) in combinations(df.columns.to_list(),2):
        print(i,j,ci_test(i,j,[]))
    print(tetrad_score3('X1', 'X2', 'X3', 'X4'))
