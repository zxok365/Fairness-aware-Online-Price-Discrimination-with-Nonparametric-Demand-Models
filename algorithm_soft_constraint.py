import numpy as np
import pmodel
import idle_algorithm
import os

Cx = 1
Cy = 1 / 200
Cz = 4


def explore_unconstrained_opt_tgt(A: pmodel.PriceModel, p_l: float, p_r: float) -> (float, float, float, float):

    r = 0
    #tmp = 25. * pow(pmodel.K, 4) * pow(pmodel.p_max, 2) / pow(pmodel.C, 2)

    tmp = 1
    while p_r - p_l > Cx * (pmodel.p_max - pmodel.p_min) * pow(A.T, -0.2):
        r = r + 1
        p_1 =  p_l * 2./3 + p_r * 1./ 3
        p_2 =  p_l * 1./3 + p_r * 2./ 3
        total_round = int(Cy * pow(A.T, 4. / 5) * np.log(A.T))
        #total_round = int(tmp * pow(A.T, 4./5) * np.log(A.T))
        d_11 = d_12 = 0
        d_21 = d_22 = 0
        #print(tmp, np.pow( A.T, 4./5), np.log(A.T), total_round)
        t = 0
        r_1 = 0
        r_2 = 0
        while t < total_round:
            t += 1
            tmp_d1, tmp_d2 = A.offer_price(p_1, p_1)
            d_11 += tmp_d1
            d_21 += tmp_d2
            tmp_d1, tmp_d2 = A.offer_price(p_2, p_2)
            d_12 += tmp_d1
            d_22 += tmp_d2
        r_11 = d_11 * (p_1 - A.c)
        r_12 = d_12 * (p_2 - A.c)
        r_21 = d_21 * (p_1 - A.c)
        r_22 = d_22 * (p_2 - A.c)
        if r_11 > r_12 and r_21 > r_22:
            p_r = p_2
        elif r_11 < r_12 and r_21 < r_22:
            p_l = p_1
        elif r_11 > r_12 and r_21 < r_22:
            return p_l, p_2, p_1, p_r
        else:
            return p_1, p_r, p_l, p_2
    return (p_l + p_r) / 2, (p_l + p_r) / 2, (p_l + p_r) / 2, (p_l + p_r) / 2


def explore_unconstrained_opt(A: pmodel.PriceModel, i: int, p_l: float, p_r: float) -> float:
    r = 0
    #tmp = 25. * pow(pmodel.K, 4) * pow(pmodel.p_max, 2) / pow(pmodel.C, 2)

    tmp = 1
    while p_r - p_l > Cx * (pmodel.p_max - pmodel.p_min) * pow(A.T, -0.2):
        r = r + 1
        p_1 =  p_l * 2./3 + p_r * 1./ 3
        p_2 =  p_l * 1./3 + p_r * 2./ 3
        total_round = int(Cy * pow(A.T, 4. / 5)) * np.log(A.T)
        tmp_round = 10
        #total_round = int(tmp * pow(A.T, 4./5) * np.log(A.T))
        d_1 = d_2 = 0
        #print(tmp, np.pow( A.T, 4./5), np.log(A.T), total_round)
        t = 0
        r_1 = 0
        r_2 = 0
        while t < total_round:
            t += 1
            tmp_d1, tmp_d2 = A.offer_price(p_1, p_1)
            if i == 1:
                d_1 += tmp_d1
            else:
                d_1 += tmp_d2
            tmp_d1, tmp_d2 = A.offer_price(p_2, p_2)
            if i == 1:
                d_2 += tmp_d1
            else:
                d_2 += tmp_d2
            if t == tmp_round:
                tmp_round *= 2
                if np.abs(d_1 * (p_1 - A.c) - d_2 * (p_2 - A.c)) > t * np.sqrt(4 / t) * np.log(A.T) * (p_2 - A.c):
                    break
        if d_1 * (p_1 - A.c) > d_2 * (p_2 - A.c):
            p_r = p_2
        else:
            p_l = p_1
    return (p_l + p_r) / 2


def explore_constrained_opt_general(A: pmodel.PriceModel, p_sharp_1: float, p_sharp_2: float) -> (float, float):

    xi = max(np.abs(p_sharp_2 - p_sharp_1), 0)
    J = int(pow(A.T, 1/5) * (pmodel.p_max - pmodel.p_min))
    delta = pmodel.lbda * xi / 2
    p_min = pmodel.p_min
    p_max = pmodel.p_max

    #pp_min = pmodel.p_min
    #pp_max = pmodel.p_max
    pp_min = max(min(p_sharp_1, p_sharp_2) - pow(A.T, -0.2), pmodel.p_min)
    pp_max = min(max(p_sharp_1, p_sharp_2) + pow(A.T, -0.2), pmodel.p_max)

    if xi == 0:
        print("xi == 0:", pp_min, pp_max)
        p = idle_algorithm.search_best_reward(A, pp_min, pp_max)
        return p, p
    if p_min > p_max:
        p_min = p_max
        print("Error, found strange stuff during constrained_OPT algorithm!")

    p_1 = 0
    p_2 = 0
    r1 = [0] * (J + 1)
    r2 = [0] * (J + 1)
    m1 = [0] * (J + 1)
    m2 = [0] * (J + 1)
    jmin = J + 1
    jmax = 0
    delta = (p_max - p_min) / J
    for j in range(0, J + 1):
        d_1 = 0
        d_2 = 0
        l_j = p_min + j * delta
 #       if l_j < pp_min or l_j > pp_max:
            # print(pp_min, pp_max, l_j)
 #           continue
        if l_j > pp_min:
            jmin = min(jmin, j)
        if l_j < pp_max:
            jmax = max(jmax, j)
        total_round = Cz * int(pow(A.T, 2/5) * np.log(A.T))
        for t in range(1, total_round + 1):
            tmp_d1, tmp_d2, tm1, tm2 = A.offer_price_soft(l_j, l_j)
            m1[j] += tm1
            m2[j] += tm2
            r1[j] += tmp_d1
            r2[j] += tmp_d2
        r1[j] = r1[j] / total_round * (l_j - A.c)
        r2[j] = r2[j] / total_round * (l_j - A.c)
        m1[j] /= total_round
        m2[j] /= total_round
    max_value = 0
    if p_sharp_1 > p_sharp_2:
        pj1 = jmax
        pj2 = jmin
    else:
        pj1 = jmin
        pj2 = jmax

    #print(p_min + pj1 / J * (p_max - p_min), p_min + pj2 / J * (p_max - p_min), m1[pj1], m2[pj2])

    for j_1 in range(jmin, jmax + 1):
        for j_2 in range(jmin, jmax + 1):
            tmp_value = r1[j_1] + r2[j_2] - pmodel.gamma * max(np.abs(m1[j_1] - m2[j_2]) - pmodel.lbda * np.abs(m1[pj1] - m2[pj2]), 0)
            if tmp_value > max_value:
                max_value = tmp_value
                p_1 = p_min + j_1 * delta
                p_2 = p_min + j_2 * delta

    return p_1, p_2


def fairness_aware_dynamic_pricing(A: pmodel.PriceModel) -> (float, float):
    l1, r1, l2, r2 = explore_unconstrained_opt_tgt(A, pmodel.p_min, pmodel.p_max)
    p_sharp_1 = explore_unconstrained_opt(A, 1, l1, r1)
    p_sharp_2 = explore_unconstrained_opt(A, 2, l2, r2)
    print("Soft-Algorithm best sol: ", p_sharp_1, p_sharp_2)
    print("Soft-Step1 complete, Current Time:", A.t)
    print("Soft-Step1 complete, Current Regret", A.regret)
    p_1, p_2 = explore_constrained_opt_general(A, p_sharp_1, p_sharp_2)
    print("Soft-Step2 complete, Current Time:", A.t)
    print("Soft-Step2 complete, Current Regret", A.regret)
    for t in range(A.t, A.T):
        A.offer_price(p_1, p_2)
    return p_1, p_2

