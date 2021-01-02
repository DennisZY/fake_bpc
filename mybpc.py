from utils import gauss_ci_test_generator, gen_tetrad_func

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
# import logging
import pandas as pd
import numpy as np
from collections import deque


def build_pure_clusters(df, test_generator=gauss_ci_test_generator, independence_alpha=0.1, tetrad_alpha=0.01):
    independence_test = test_generator(df, independence_alpha)
    tetrad_score1, tetrad_score3, tetrad_holds = gen_tetrad_func(df, tetrad_alpha)
    v_labels = df.columns.to_list()

