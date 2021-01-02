import numpy as np
from scipy.stats import norm
from math import log1p, sqrt
import logging
import time
import functools

def gauss_ci_test_generator(df, alpha):
    cov = df.cov()
    n = df.shape[0]

    def gauss_ci_test(x, y, s):
        nonlocal cov
        nonlocal n
        if len(s) == 0:
            r = cov.loc[x][y]
        elif len(s) == 1:
            # print(x,y,s)
            # print(corr)
            # print(corr.index.to_list())
            # print(corr.columns.to_list())
            # print(corr[s,y])
            r = (cov.loc[x, y] - cov.loc[x, s] * cov.loc[s, y]) / (
                (sqrt(1 - cov.loc[x, s] ** 2) * sqrt(1 - cov.loc[s, y] ** 2)))
        else:
            li = [x] + [y] + s
            m = cov.loc[li, li]
            pm = np.linalg.pinv(m)
            r = -1.0 * pm[0, 1] / sqrt(abs(pm[0, 0] * pm[1, 1]))
        cut_at = 0.9999999
        r = min(cut_at, max(-cut_at, r.item()))
        res = sqrt(n - len(s) - 3) * .5 * log1p((2 * r) / (1 - r))
        return 2 * (1 - norm.cdf(abs(res))) >= alpha
        # return res < norm.ppf(1.0 - alpha / 2.0)
    return gauss_ci_test


def gen_tetrad_func(df, alpha):
    cov = df.cov()
    n = df.shape[0]

    def wishart_test_tetrad_difference(a0, a1, a2, a3):
        nonlocal cov
        nonlocal n
        product1 = cov.loc[a0, a0] * cov.loc[a3, a3] - cov.loc[a0, a3] * cov.loc[a0, a3]
        product2 = cov.loc[a1, a1] * cov.loc[a2, a2] - cov.loc[a1, a2] * cov.loc[a1, a2]
        product3 = (n + 1) / ((n - 1) * (n - 2)) * product1 * product2
        li = [a0, a1, a2, a3]
        m = cov.loc[li, li]
        determinant = np.linalg.det(m)
        var = (product3 - determinant / (n - 2))
        return sqrt(abs(var))

    def wishart_eval_tetrad_difference(i, j, k, l):
        nonlocal cov
        nonlocal n
        tau_ijkl = cov.loc[i, j] * cov.loc[k, l] - cov.loc[i, k] * cov.loc[j, l]

        sd = wishart_test_tetrad_difference(i, j, k, l)
        ratio = tau_ijkl / sd

        return 2.0 * norm.cdf(abs(ratio))

    def wishart_eval_tetrad_difference2(i, j, k, l):
        nonlocal cov
        nonlocal n
        nonlocal alpha
        return wishart_eval_tetrad_difference(i, j, k, l) > alpha and wishart_eval_tetrad_difference(i, j, l, k) > alpha

    def tetrad_holds(v1, v2, v3, v4) -> bool:
        nonlocal alpha
        res = wishart_eval_tetrad_difference(v1, v2, v3, v4)
        return res >= alpha

    def tetrad_score1(v1, v2, v3, v4) -> bool:
        return tetrad_holds(v1, v3, v4, v2) \
               and not tetrad_holds(v1, v3, v2, v4) \
               and not tetrad_holds(v1, v4, v2, v3)

    def tetrad_score(v1, v2, v3, v4):
        if wishart_eval_tetrad_difference2(v1, v2, v3, v4):
            return 3
        else:
            return 1

    def tetrad_score3(v1, v2, v3, v4) -> bool:
        return tetrad_score(v1, v3, v4, v2) == 3

    return tetrad_score1, tetrad_score3, tetrad_holds



# https://wiki.python.org/moin/PythonDecoratorLibrary#Singleton
def singleton(cls):
    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kwargs):
        it = cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kwargs)
        it.__init_original__(*args, **kwargs)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__
    return cls


# @singleton
# class Logger(object):
#     def __new__(cls, *args, **kwargs):
#         cls.x = 10
#         return object.__new__(cls)
#
#     def __init__(self, log_file_name, log_level, logger_name):
#         # 创建一个logger
#         self.__logger = logging.getLogger(logger_name)
#
#         # 指定日志的最低输出级别，默认为WARN级别
#         self.__logger.setLevel(log_level)
#
#         # 创建一个handler用于写入日志文件
#         file_handler = logging.FileHandler(log_file_name)
#
#         # 创建一个handler用于输出控制台
#         console_handler = logging.StreamHandler()
#
#         # 定义handler的输出格式
#         formatter = logging.Formatter(
#             '[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
#
#         # 给logger添加handler
#         self.__logger.addHandler(file_handler)
#         self.__logger.addHandler(console_handler)
#
#     def get_log(self):
#         return self.__logger
