from data import f11_random_test_data, gen_data_set
from bpc import build_pure_clusters

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
# import logging
# import time


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
    #                     filename=str.format('log-%s.log',time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
    #                     filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    #                     # a是追加模式，默认如果不写的话，就是追加模式
    #                     format=
    #                     '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    #                     # 日志格式
    #                     )
    # g = build_pure_clusters(gen_data_set(sm='sm3',mm='mm3',sample_size=10000))
    # df = pd.read_csv()
    # data = pd.read_csv('test10000-3-9.txt', sep="\t")
    # print(data)
    # g = build_pure_clusters(data)
    g = build_pure_clusters(f11_random_test_data())
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos, with_labels=True)
    plt.show()
