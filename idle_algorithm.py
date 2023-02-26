import numpy as np
import pmodel

eps = 1e-7


def search_best_reward(A: pmodel.PriceModel, p_min : float, p_max : float) -> float:
    p_l = p_min
    p_r = p_max
    d_1 = d_2 = 0
    r_1 = r_2 = 0
    while p_r - p_l > eps:
        p_1 = 2. / 3 * p_l + 1. / 3 * p_r
        p_2 = 1. / 3 * p_l + 2. / 3 * p_r
        total_round = 1 / 160 * int(pow(A.T, 4 / 5)) * int(np.log(A.T))
        t = 0
        tmp_round = 10
        d_1 = d_2 = 0
        r_1 = r_2 = 0
        while t < total_round:
            t += 1
            if A.t + 2 > A.T:
                print("idle-halt: ", p_l, p_r)
                return (p_l + p_r) / 2.
            tmpd_1, tmpd_2 = A.offer_price(p_1, p_1)
            tmpd_3, tmpd_4 = A.offer_price(p_2, p_2)
            d_1 += tmpd_1 + tmpd_2
            d_2 += tmpd_3 + tmpd_4
            r_1 = d_1 * (p_1 - A.c)
            r_2 = d_2 * (p_2 - A.c)
            if t == tmp_round:
                tmp_round *= 2
                if np.abs(r_1 - r_2) > t * np.sqrt(4 / t) * np.log(A.T):
                    break
        if r_1 < r_2:
            p_l = p_1
        else:
            p_r = p_2
    return (p_l + p_r) / 2.


def fairness_aware_dynamic_pricing(A: pmodel.PriceModel, p_min: float, p_max: float) -> (float, float):
    p_1 = search_best_reward(A, p_min, p_max)
    for t in range(A.t, A.T):
        A.offer_price(p_1, p_1)
    return p_1, p_1
