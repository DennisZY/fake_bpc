from utils import gauss_ci_test_generator, gen_tetrad_func

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
# import logging
import pandas as pd
import numpy as np
from collections import deque


def remove_no_parent_observed_node(h):
    observed_has_parent = set()
    for i in h.nodes():
        if i[0] == 'L':
            for j in h.edges(i):
                observed_has_parent |= {j[1]}
    remove_observed = []
    for i in set(h.nodes()) - observed_has_parent:
        if i[0] != 'L':
            remove_observed.append(i)
    for i in remove_observed:
        h.remove_node(i)
    print('log2', remove_observed)


def build_pure_clusters(df, test_generator=gauss_ci_test_generator, independence_alpha=0.01, tetrad_alpha=0.001):
    independence_test = test_generator(df, independence_alpha)
    tetrad_score1, tetrad_score3, tetrad_holds = gen_tetrad_func(df, tetrad_alpha)
    v_labels = df.columns.to_list()

    # Stage 0: find pattern

    # Stage 0.1: identify (partially) uncorrelated and impure pairs
    print("Stage 0.1: identify (partially) uncorrelated and impure pairs")
    g = nx.complete_graph(v_labels)
    for (i, j) in combinations(v_labels, 2):
        if g.has_edge(i, j) and independence_test(i, j, []):
            g.remove_edge(i, j)

    # for (i, j) in combinations(v_labels, 2):
    #     if g.has_edge(i, j):
    #         for k in v_labels:
    #             if i == k or j == k:
    #                 continue
    #             if independence_test(i, j, [k]):
    #                 g.remove_edge(i, j)
    #                 break

    # Stage 0.2: test constriant set
    print("Stage 0.2: test constriant set")
    for (a, b, c, d, e, f) in combinations(v_labels, 6):
        correlate = True
        for (i, j) in combinations([a, b, c, d, e, f], 2):
            if not g.has_edge(i, j):
                correlate = False
                break
        if not correlate:
            continue

        for (x1, x2, x3, y1, y2, y3) in permutations([a, b, c, d, e, f], 6):
            # CS1
            if g.has_edge(x1, y1) and tetrad_score3(x1, y1, x2, x3) and tetrad_score3(y1, x1, y2, y3) \
                    and not tetrad_holds(x1, x2, y2, y1):
                g.remove_edge(x1, y1)
                print('CS1', x1, x2, x3, y1, y2, y3)
                # logging.debug('{} {} {} {} {} {} {}'.format('CS1',x1,x2,x3,y1,y2,y3))
            # CS2
            if g.has_edge(x1, y1) and tetrad_holds(x1, y1, y2, x2) and tetrad_holds(x2, y1, y3, y2) \
                    and tetrad_holds(x1, x2, y2, x3) and not tetrad_holds(x1, x2, y2, y1):
                g.remove_edge(x1, y1)
                print('CS2', x1, x2, x3, y1, y2, y3)

            # CS3
            if g.has_edge(x1, y1) and tetrad_score3(x1, y1, y2, y3) and tetrad_score3(x1, y2, x2, x3) \
                    and tetrad_score3(x1, y3, x2, x3) and not tetrad_holds(x1, x2, y2, y3):
                g.remove_edge(x1, y1)
                print('CS3', x1, x2, x3, y1, y2, y3)

    # Stage 0.3: initial graph H
    print("Stage 0.3: initial graph H")
    h = nx.MultiDiGraph()
    for i in v_labels:
        h.add_node(i)

    # Stage 0.4: find maximal cliques
    # def find_components():
    #     node_set = set(v_labels)
    #     while len(node_set) != 0:
    print("Stage 0.4: find maximal cliques")
    res = nx.find_cliques(g)
    new_latent_id = 1
    for clique in res:
        latent_name = 'L' + str(new_latent_id)
        new_latent_id += 1
        h.add_node(latent_name)
        for i in clique:
            h.add_edge(latent_name, i)

    # Stage 0.5: add undirected edge
    print("Stage 0.5: add undirected edge")
    condition_test = pd.DataFrame(True, index=v_labels, columns=v_labels, dtype=bool)
    for i in v_labels:
        condition_test.loc[i, i] = False
    for (a, b) in combinations(v_labels, 2):
        if not g.has_edge(a, b):
            continue
        for (c, d) in combinations(v_labels, 2):
            if a == c or a == d or b == c or b == d:
                continue
            flag = True
            for (i, j) in combinations([a, b, c, d], 2):
                if not g.has_edge(i, j):
                    flag = False
                    break
            if not flag or tetrad_score3(a, b, c, d):
                condition_test[a][b] = False
        if condition_test[a][b]:
            h.add_edge(a, b)
            h.add_edge(b, a)

    pos = nx.circular_layout(h)
    nx.draw_networkx(h, pos, with_labels=True)
    plt.show()

    # Stage 1.1: remove the latent which clusters of size 1
    print("Stage 1.1: remove the latent which clusters of size 1")
    remove_latent = []
    for i in h.nodes():
        if i[0] == 'L' and len(h.edges(i)) == 1:
            remove_latent.append(i)
    for i in remove_latent:
        h.remove_node(i)
    print('log1', remove_latent)

    # Stage 1.2: remove the observed node that hasn't latent parent
    print("Stage 1.2: remove the observed node that hasn't latent parent")
    remove_no_parent_observed_node(h)

    # Stage 2: remove the observed node that have more than one latent parent
    print("Stage 2: remove the observed node that have more than one latent parent")
    latent_parent_cnt = np.zeros(1, dtype=[(i, int) for i in h.nodes()])
    for i in h.nodes():
        if i[0] == 'L':
            for j in h.edges(i):
                # print(j)
                latent_parent_cnt[0][j[1]] += 1
    remove_observed = []
    for i in h.nodes():
        if latent_parent_cnt[0][i] > 1:
            remove_observed.append(i)
    for i in remove_observed:
        h.remove_node(i)
    print('log3', remove_observed)

    # Stage 3: for all pairs of nodes linked by an undirected edge, choose one element of each pair to be removed.
    print("Stage 3: for all pairs of nodes linked by an undirected edge, choose one element of each pair to be removed.")
    tmp_graph = nx.Graph()
    for i in h.edges():
        if h.has_edge(i[0], i[1]) and h.has_edge(i[1], i[0]):
            tmp_graph.add_edge(i[0], i[1])
    q = deque()
    remove_observed = []
    while True:
        cont = False
        for i in tmp_graph.nodes():
            cont = True
            q.append((i, False))
            while len(q) != 0:
                top = q.popleft()
                print(top)
                for j in tmp_graph.edges(i):
                    q.append((j[1], not top[1]))
                if top[1]:
                    remove_observed.append(top[0])
                tmp_graph.remove_node(top[0])
            break
        if not cont:
            break
    for i in remove_observed:
        h.remove_node(i)

    # Stage 4: for all pairs of nodes linked by an undirected edge, choose one element of each pair to be removed.
    print("Stage 4: for all pairs of nodes linked by an undirected edge, choose one element of each pair to be removed.")
    latent_nodes = [i for i in h.nodes() if i[0] == 'L']
    for i in latent_nodes:
        if len(h.edges(i)) < 3:
            continue
        node_set = [x for (_, x) in h.edges(i)]
        for (a, b, c) in combinations(node_set, 3):
            remove_observed = []
            for j in h.nodes():
                if j[0] == 'L':
                    continue
                if a == j or b == j or c == j:
                    continue
                if not tetrad_score3(a, b, c, j):
                    remove_observed.append(j)
            for j in remove_observed:
                h.remove_node(j)

    # Stage 5-8: to be continued

    # Stage 9: Remove all latents with less than three children, and their respective measures
    print("Stage 9: Remove all latents with less than three children, and their respective measures")
    latent_nodes = [i for i in h.nodes() if i[0] == 'L']
    for i in latent_nodes:
        if len(h.edges(i)) < 3:
            h.remove_node(i)
    remove_no_parent_observed_node(h)
    observed_node_cnt = 0
    for i in h.nodes():
        if i[0] != 'L':
            observed_node_cnt += 1
    if observed_node_cnt < 4:
        return nx.empty_graph()
    return h
