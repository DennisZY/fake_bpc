import numpy as np
import pandas as pd
from collections import deque


def f11_random_test_data():
    # np.random.seed(0)
    l_size = 3
    o_size = 9
    sample_size = 1000
    v_labels = ['L' + str(i + 1) for i in range(l_size)] + ['X' + str(i + 1) for i in range(o_size)]
    has_edge = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)

    has_edge.loc['L2', 'L1'] = True
    has_edge.loc['L2', 'L3'] = True

    has_edge.loc['L1', 'X1'] = True
    has_edge.loc['L1', 'X2'] = True
    has_edge.loc['L1', 'X3'] = True
    has_edge.loc['L2', 'X4'] = True
    has_edge.loc['L2', 'X5'] = True
    has_edge.loc['L2', 'X6'] = True
    has_edge.loc['L3', 'X7'] = True
    has_edge.loc['L3', 'X8'] = True
    has_edge.loc['L3', 'X9'] = True

    # has_edge.loc['L3', 'X9'] = True
    # has_edge.loc['X5', 'X2'] = True

    # l_size = 2
    # o_size = 12
    # sample_size = 10000
    # v_labels = ['L' + str(i + 1) for i in range(l_size)] + ['X' + str(i + 1) for i in range(o_size)]
    # has_edge = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)
    #
    # has_edge.loc['L1'].at['L2'] = True
    # has_edge.loc['L1'].at['X1'] = True
    # has_edge.loc['L1'].at['X2'] = True
    # has_edge.loc['L1'].at['X3'] = True
    # has_edge.loc['L1'].at['X7'] = True
    # has_edge.loc['L1'].at['X8'] = True
    # has_edge.loc['L1'].at['X11'] = True
    # has_edge.loc['L2'].at['X4'] = True
    # has_edge.loc['L2'].at['X5'] = True
    # has_edge.loc['L2'].at['X6'] = True
    # has_edge.loc['L2'].at['X8'] = True
    # has_edge.loc['L2'].at['X9'] = True
    # has_edge.loc['L2'].at['X10'] = True
    # has_edge.loc['L2'].at['X12'] = True
    # has_edge.loc['X7'].at['X8'] = True
    # has_edge.loc['X9'].at['X10'] = True
    # has_edge.loc['X11'].at['X12'] = True

    in_degree = [0] * (l_size + o_size)

    for i in v_labels:
        for ind, j in enumerate(v_labels):
            if has_edge.loc[i, j]:
                in_degree[ind] += 1
    df = pd.DataFrame(np.concatenate((np.random.normal(0, 3, size=(sample_size, l_size)),
                                      np.random.normal(0, 3, size=(sample_size, o_size))), axis=1),
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
    return df.loc[:, 'X1':]


def gen_data_set(sm='sm1', mm='mm1', sample_size=200):
    l_size = 0
    o_size = 0
    if sm == 'sm1':
        l_size = 3
    elif sm == 'sm2':
        l_size = 3
    elif sm == 'sm3':
        l_size = 4
    else:
        exit(1)
    if mm == 'mm1':
        o_size = 9
    elif mm == 'mm2':
        o_size = 15
    elif mm == 'mm3':
        o_size = 18
    else:
        exit(1)
    v_labels = ['L' + str(i + 1) for i in range(l_size)] + ['X' + str(i + 1) for i in range(o_size)]
    has_edge = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)
    if sm == 'sm1':
        has_edge.loc['L1', 'L2'] = True
        has_edge.loc['L3', 'L2'] = True
    elif sm == 'sm2':
        has_edge.loc['L1', 'L2'] = True
        has_edge.loc['L2', 'L3'] = True
    elif sm == 'sm3':
        has_edge.loc['L1', 'L2'] = True
        has_edge.loc['L2', 'L3'] = True
        has_edge.loc['L1', 'L4'] = True
        has_edge.loc['L4', 'L3'] = True
    else:
        exit(1)
    if mm == 'mm1':
        has_edge.loc['L1', 'X1'] = True
        has_edge.loc['L1', 'X2'] = True
        has_edge.loc['L1', 'X3'] = True
        has_edge.loc['L2', 'X4'] = True
        has_edge.loc['L2', 'X5'] = True
        has_edge.loc['L2', 'X6'] = True
        has_edge.loc['L3', 'X7'] = True
        has_edge.loc['L3', 'X8'] = True
        has_edge.loc['L3', 'X9'] = True
    elif mm == 'mm2':
        has_edge.loc['L1', 'X1'] = True
        has_edge.loc['L1', 'X2'] = True
        has_edge.loc['L1', 'X3'] = True
        has_edge.loc['L2', 'X4'] = True
        has_edge.loc['L2', 'X5'] = True
        has_edge.loc['L2', 'X6'] = True
        has_edge.loc['L3', 'X7'] = True
        has_edge.loc['L3', 'X8'] = True
        has_edge.loc['L3', 'X9'] = True

        has_edge.loc['L1', 'X10'] = True
        has_edge.loc['L1', 'X11'] = True
        has_edge.loc['X10', 'X11'] = True
        has_edge.loc['L2', 'X11'] = True
        has_edge.loc['L2', 'X12'] = True
        has_edge.loc['L2', 'X13'] = True
        has_edge.loc['X12', 'X13'] = True
        has_edge.loc['L3', 'X13'] = True
        has_edge.loc['L3', 'X14'] = True
        has_edge.loc['L3', 'X15'] = True
        has_edge.loc['X14', 'X15'] = True


    elif mm == 'mm3':
        has_edge.loc['L1', 'X1'] = True
        has_edge.loc['L1', 'X2'] = True
        has_edge.loc['L1', 'X3'] = True
        has_edge.loc['L2', 'X4'] = True
        has_edge.loc['L2', 'X5'] = True
        has_edge.loc['L2', 'X6'] = True
        has_edge.loc['L3', 'X7'] = True
        has_edge.loc['L3', 'X8'] = True
        has_edge.loc['L3', 'X9'] = True

        has_edge.loc['L1', 'X10'] = True
        has_edge.loc['L1', 'X11'] = True
        has_edge.loc['X10', 'X11'] = True
        has_edge.loc['L2', 'X11'] = True
        has_edge.loc['L2', 'X12'] = True
        has_edge.loc['L2', 'X13'] = True
        has_edge.loc['X12', 'X13'] = True
        has_edge.loc['L3', 'X13'] = True
        has_edge.loc['L3', 'X14'] = True
        has_edge.loc['L3', 'X15'] = True
        has_edge.loc['X14', 'X15'] = True

        has_edge.loc['L1', 'X16'] = True
        has_edge.loc['L2', 'X17'] = True
        has_edge.loc['L3', 'X18'] = True
        has_edge.loc['X16', 'X17'] = True
        has_edge.loc['X17', 'X18'] = True

    else:
        exit(1)

    in_degree = [0] * (l_size + o_size)

    for i in v_labels:
        for ind, j in enumerate(v_labels):
            if has_edge.loc[i, j]:
                in_degree[ind] += 1
    df = pd.DataFrame(np.random.normal(0, np.random.uniform(1, 3), size=(sample_size, l_size + o_size)),
                      index=range(sample_size), columns=v_labels, dtype=float)
    q = deque()
    for ind, val in enumerate(in_degree):
        if val == 0:
            q.append(v_labels[ind])

    def gen_coef():
        if np.random.uniform(-1, 1) > .0:
            return np.random.uniform(0.5, 1.5)
        else:
            return np.random.uniform(-1.5, -0.5)

    while len(q) != 0:
        top = q.popleft()
        for ind, i in enumerate(has_edge.loc[top, :]):
            if i:
                df.loc[:, v_labels[ind]] += gen_coef() * df.loc[:, top]
                in_degree[ind] -= 1
                if in_degree[ind] == 0:
                    q.append(v_labels[ind])
    return df.loc[:, 'X1':]


if __name__ == '__main__':
    print(f11_random_test_data())
