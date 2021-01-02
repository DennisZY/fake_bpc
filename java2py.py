from utils import gauss_ci_test_generator, gen_tetrad_func

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
# import logging
import pandas as pd
import numpy as np
from collections import deque

EDGE_NONE = 0
EDGE_BLACK = 1
EDGE_GRAY = 2
EDGE_BLUE = 3
EDGE_YELLOW = 4
EDGE_RED = 4

def find_maximal_cliques(component, ng):
    g = nx.Graph()
    for i in component:
        for j in component:
            if i == j:
                continue
            if ng.loc[i,j] != EDGE_NONE:
                g.add_edge(i,j)
    res = nx.find_cliques()
    return list(res)

def find_components(graph, color):
    v_labels = graph.columns.to_list()
    marked = np.array(False, dtype=[(i, bool) for i in v_labels])
    num_marked = 0
    output = []
    temp_component = [0 for _ in v_labels]
    while num_marked != len(v_labels):
        size_temp = 0
        while True:
            no_change = True
            for (ind, mark) in enumerate(marked):
                if mark:
                    continue
                in_component = False
                for j in range(size_temp):
                    if in_component:
                        break
                    if graph.loc[v_labels[ind]][v_labels[temp_component[j]]] == color:
                        in_component = True
                if size_temp == 0 or in_component:
                    temp_component[size_temp] = ind
                    size_temp += 1
                    marked[ind] = True
                    no_change = False
                    num_marked += 1
            if no_change:
                break
        if size_temp > 1:
            new_partition = temp_component[:size_temp]
            output.append(new_partition)
    return output


def initial_measurement_pattern(df, ng, cv):
    v_labels = df.columns.to_list()
    not_yellow = pd.DataFrame(False, index=v_labels, columns=v_labels, dtype=bool)
    uncorrelated = gauss_ci_test_generator(df, 0.01)
    tetrad_score1, tetrad_score3, tetrad_holds = gen_tetrad_func(df, 0.001)
    # Stage 1: identify (partially) uncorrelated and impure pairs
    for (v1, v2) in combinations(v_labels, 2):
        ng.loc[v1, v2] = ng.loc[v2, v1] = EDGE_BLACK
    for (v1, v2) in combinations(v_labels, 2):
        if uncorrelated(v1, v2, []):
            cv.loc[v1, v2] = cv.loc[v2, v1] = EDGE_NONE
        else:
            cv.loc[v1, v2] = cv.loc[v2, v1] = EDGE_BLACK
        ng.loc[v1, v2] = ng.loc[v2, v1] = cv.loc[v1, v2]

    for (v1, v2) in combinations(v_labels, 2):
        if cv.loc[v1, v2] == EDGE_NONE:
            continue
        for v3 in v_labels:
            if v1 == v3 or v2 == v3:
                continue
            if uncorrelated(v1, v2, [v3]):
                cv.loc[v1, v2] = cv.loc[v2, v1] = EDGE_NONE
                break

    for (v1, v2) in combinations(v_labels, 2):
        if ng.loc[v1, v2] != EDGE_BLACK:
            continue
        not_found = True
        for (i, v3) in enumerate(v_labels):
            if not not_found:
                break
            if v1 == v3 or v2 == v3 or \
                    ng.loc[v1, v3] == EDGE_NONE or ng.loc[v1, v3] == EDGE_GRAY or \
                    ng.loc[v2, v3] == EDGE_NONE or ng.loc[v2, v3] == EDGE_GRAY:
                continue
            for v4 in v_labels[i + 1:]:
                if not not_found:
                    break
                if v1 == v4 or v2 == v4 or \
                        ng.loc[v1, v4] == EDGE_NONE or ng.loc[v1, v4] == EDGE_GRAY or \
                        ng.loc[v2, v4] == EDGE_NONE or ng.loc[v2, v4] == EDGE_GRAY or \
                        ng.loc[v3, v4] == EDGE_NONE or ng.loc[v3, v4] == EDGE_GRAY:
                    continue
                if tetrad_score3(v1, v2, v3, v4):
                    not_found = False
                    ng.loc[v1, v2] = ng.loc[v2, v1] = EDGE_BLUE
                    ng.loc[v1, v3] = ng.loc[v3, v1] = EDGE_BLUE
                    ng.loc[v1, v4] = ng.loc[v4, v1] = EDGE_BLUE
                    ng.loc[v2, v3] = ng.loc[v3, v2] = EDGE_BLUE
                    ng.loc[v2, v4] = ng.loc[v4, v2] = EDGE_BLUE
                    ng.loc[v3, v4] = ng.loc[v4, v3] = EDGE_BLUE
        if not_found:
            ng.loc[v1, v2] = ng.loc[v2, v1] = EDGE_GRAY

    def unclustered_partial1(v1, v2, v3, v4):
        return tetrad_score3(v1, v2, v3, v4)

    def unclustered_partial2(v1, v2, v3, v4, v5):
        return tetrad_score3(v1, v2, v3, v5) and tetrad_score1(v1, v2, v4, v5) and \
               tetrad_score1(v2, v3, v4, v5) and tetrad_score1(v1, v3, v4, v5)

    def unclustered_partial3(v1, v2, v3, v4, v5, v6):
        return tetrad_score3(v1, v2, v3, v6) and tetrad_score3(v4, v5, v6, v1) and \
               tetrad_score3(v4, v5, v6, v2) and tetrad_score3(v4, v5, v6, v3) and \
               tetrad_score1(v1, v2, v4, v6) and tetrad_score1(v1, v2, v5, v6) and \
               tetrad_score1(v2, v3, v4, v6) and tetrad_score1(v2, v3, v5, v6) and \
               tetrad_score1(v1, v3, v4, v6) and tetrad_score1(v1, v3, v5, v5)

    for (v1, v2) in combinations(v_labels, 2):
        if ng.loc[v1, v2] != EDGE_BLUE:
            continue
        not_found = True
        for (i, v3) in enumerate(v_labels):
            if not not_found:
                break
            if v1 == v3 or v2 == v3 or \
                    ng.loc[v1, v3] == EDGE_GRAY or cv.loc[v1, v3] != EDGE_BLACK or \
                    ng.loc[v2, v3] == EDGE_GRAY or cv.loc[v2, v3] != EDGE_BLACK:
                continue
            for v5 in v_labels[i + 1:]:
                if not not_found:
                    break
                if v1 == v5 or v2 == v5 or \
                        ng.loc[v1, v5] == EDGE_GRAY or cv.loc[v1, v5] != EDGE_BLACK or \
                        ng.loc[v2, v5] == EDGE_GRAY or cv.loc[v2, v5] != EDGE_BLACK or \
                        ng.loc[v3, v5] == EDGE_GRAY or cv.loc[v3, v5] != EDGE_BLACK or \
                        not unclustered_partial1(v1, v3, v5, v2):
                    continue
                for (j, v4) in enumerate(v_labels):
                    if not not_found:
                        break
                    if v1 == v4 or v2 == v4 or v3 == v4 or v5 == v4 or \
                            ng.loc[v1, v4] == EDGE_GRAY or cv.loc[v1, v4] != EDGE_BLACK or \
                            ng.loc[v2, v4] == EDGE_GRAY or cv.loc[v2, v4] != EDGE_BLACK or \
                            ng.loc[v3, v4] == EDGE_GRAY or cv.loc[v3, v4] != EDGE_BLACK or \
                            ng.loc[v5, v4] == EDGE_GRAY or cv.loc[v5, v4] != EDGE_BLACK or \
                            not unclustered_partial2(v1, v3, v5, v2, v4):
                        continue
                    for v6 in v_labels[j + 1:]:
                        if not not_found:
                            break
                        if v1 == v6 or v2 == v6 or v3 == v6 or v5 == v6 or \
                                ng.loc[v1, v6] == EDGE_GRAY or cv.loc[v1, v6] != EDGE_BLACK or \
                                ng.loc[v2, v6] == EDGE_GRAY or cv.loc[v2, v6] != EDGE_BLACK or \
                                ng.loc[v3, v6] == EDGE_GRAY or cv.loc[v3, v6] != EDGE_BLACK or \
                                ng.loc[v4, v6] == EDGE_GRAY or cv.loc[v5, v6] != EDGE_BLACK or \
                                ng.loc[v5, v6] == EDGE_GRAY or cv.loc[v5, v6] != EDGE_BLACK or \
                                not unclustered_partial3(v1, v3, v5, v2, v4, v6):
                            continue
                        not_found = False
                        ng.loc[v1, v2] = ng.loc[v2, v1] = EDGE_NONE
                        ng.loc[v1, v4] = ng.loc[v4, v1] = EDGE_NONE
                        ng.loc[v1, v6] = ng.loc[v6, v1] = EDGE_NONE
                        ng.loc[v3, v2] = ng.loc[v2, v3] = EDGE_NONE
                        ng.loc[v3, v4] = ng.loc[v4, v3] = EDGE_NONE
                        ng.loc[v3, v6] = ng.loc[v6, v3] = EDGE_NONE
                        ng.loc[v5, v2] = ng.loc[v2, v5] = EDGE_NONE
                        ng.loc[v5, v4] = ng.loc[v4, v5] = EDGE_NONE
                        ng.loc[v5, v6] = ng.loc[v6, v5] = EDGE_NONE
                        not_yellow.loc[v1, v3] = not_yellow.loc[v3, v1] = True
                        not_yellow.loc[v1, v5] = not_yellow.loc[v5, v1] = True
                        not_yellow.loc[v3, v5] = not_yellow.loc[v5, v3] = True
                        not_yellow.loc[v2, v4] = not_yellow.loc[v4, v2] = True
                        not_yellow.loc[v2, v6] = not_yellow.loc[v6, v2] = True
                        not_yellow.loc[v4, v6] = not_yellow.loc[v6, v4] = True
        if not_yellow.loc[v1, v2]:
            not_found = False
        if not_found:
            # Trying to find unclustered({v1, v2, v3}, {v4, v5, v6})
            for v3 in v_labels:
                if not not_found:
                    break
                if v1 == v3 or v2 == v3 or \
                        ng.loc[v1, v3] == EDGE_GRAY or cv.loc[v1, v3] != EDGE_BLACK or \
                        ng.loc[v2, v3] == EDGE_GRAY or cv.loc[v2, v3] != EDGE_BLACK:
                    continue
                for (i, v4) in enumerate(v_labels):
                    if not not_found:
                        break
                    if v1 == v4 or v2 == v4 or v3 == v4 or \
                            ng.loc[v1, v4] == EDGE_GRAY or cv.loc[v1, v4] != EDGE_BLACK or \
                            ng.loc[v2, v4] == EDGE_GRAY or cv.loc[v2, v4] != EDGE_BLACK or \
                            ng.loc[v3, v4] == EDGE_GRAY or cv.loc[v3, v4] != EDGE_BLACK or \
                            not unclustered_partial1(v1, v2, v3, v4):
                        continue
                    for (j, v5) in enumerate(v_labels[i + 1:]):
                        if not not_found:
                            break
                        if v1 == v5 or v2 == v5 or v3 == v5 or \
                                ng.loc[v1, v5] == EDGE_GRAY or cv.loc[v1, v5] != EDGE_BLACK or \
                                ng.loc[v2, v5] == EDGE_GRAY or cv.loc[v2, v5] != EDGE_BLACK or \
                                ng.loc[v3, v5] == EDGE_GRAY or cv.loc[v3, v5] != EDGE_BLACK or \
                                ng.loc[v4, v5] == EDGE_GRAY or cv.loc[v4, v5] != EDGE_BLACK or \
                                not unclustered_partial2(v1, v2, v3, v4, v5):
                            continue
                        for v6 in v_labels[i + j + 1:]:
                            if not not_found:
                                break
                            if v1 == v6 or v2 == v6 or v3 == v6 or \
                                    ng.loc[v1, v6] == EDGE_GRAY or cv.loc[v1, v6] != EDGE_BLACK or \
                                    ng.loc[v2, v6] == EDGE_GRAY or cv.loc[v2, v6] != EDGE_BLACK or \
                                    ng.loc[v3, v6] == EDGE_GRAY or cv.loc[v3, v6] != EDGE_BLACK or \
                                    ng.loc[v4, v6] == EDGE_GRAY or cv.loc[v5, v6] != EDGE_BLACK or \
                                    ng.loc[v5, v6] == EDGE_GRAY or cv.loc[v5, v6] != EDGE_BLACK or \
                                    not unclustered_partial3(v1, v2, v3, v4, v5, v6):
                                continue
                            not_found = False
                            ng.loc[v1, v4] = ng.loc[v4, v1] = EDGE_NONE
                            ng.loc[v1, v5] = ng.loc[v5, v1] = EDGE_NONE
                            ng.loc[v1, v6] = ng.loc[v6, v1] = EDGE_NONE
                            ng.loc[v2, v4] = ng.loc[v4, v2] = EDGE_NONE
                            ng.loc[v2, v5] = ng.loc[v5, v2] = EDGE_NONE
                            ng.loc[v2, v6] = ng.loc[v6, v2] = EDGE_NONE
                            ng.loc[v3, v4] = ng.loc[v4, v3] = EDGE_NONE
                            ng.loc[v3, v5] = ng.loc[v5, v3] = EDGE_NONE
                            ng.loc[v3, v6] = ng.loc[v6, v3] = EDGE_NONE
                            not_yellow.loc[v1, v2] = not_yellow.loc[v2, v1] = True
                            not_yellow.loc[v1, v3] = not_yellow.loc[v3, v1] = True
                            not_yellow.loc[v2, v3] = not_yellow.loc[v3, v2] = True
                            not_yellow.loc[v4, v5] = not_yellow.loc[v5, v4] = True
                            not_yellow.loc[v4, v6] = not_yellow.loc[v6, v4] = True
                            not_yellow.loc[v5, v6] = not_yellow.loc[v6, v5] = True
            if not_found:
                ng.loc[v1, v2] = ng.loc[v2, v1] = EDGE_YELLOW
    components = find_components(ng, EDGE_BLUE)
    clustering = []
    for component in components:
        for i in find_maximal_cliques(component,ng):
            clustering.append(i)


    return clustering


# 2421
def find_measurement_pattern(df):
    v_labels = df.columns.to_list()
    ng = pd.DataFrame(0, index=v_labels, columns=v_labels, dtype=int)
    cv = pd.DataFrame(0, index=v_labels, columns=v_labels, dtype=int)
    selected = np.array(False, dtype=[(i, bool) for i in v_labels])

    initial_clustering = initial_measurement_pattern(df, ng, cv)

    forbidden_list = []
    for c in initial_clustering:
        for (i, j) in combinations(c, 2):
            forbidden_list.append({i, j})
            selected[i] = selected[j] = True

    for (c1, c2) in combinations(initial_clustering, 2):
        for i in c1:
            for j in c2:
                forbidden_list.append({i, j})
    # 2527
    #   Stage 1: identify (partially) uncorrelated and impure pairs
    for i in v_labels:
        for j in v_labels:
            if selected[i] and selected[j] and (ng.loc[i, j] == EDGE_BLUE or ng.loc[i, j] == EDGE_YELLOW):
                ng.loc[i, j] = EDGE_RED
            elif (not selected[i] or not selected[j]) and ng.loc[i][j] == EDGE_YELLOW:
                ng.loc[i, j] = EDGE_BLUE


    return []


#   Stage 2: prune blue edges


def bpc(df):
    clustering = find_measurement_pattern(df)
    clusters = []
    for i in clustering:
        if len(i) >= 3:
            clusters.append(i)
    g = nx.empty_graph(nx.DiGraph)
    nodes = df.columns.to_list()
    for i in nodes:
        g.add_node(i)
    for (ind, clu) in enumerate(clustering):
        latent_name = 'L' + str(ind + 1)
        g.add(latent_name)
        for i in clu:
            g.add_edge(latent_name, i)
    return g
