import numpy as np
import pmodel
import idle_algorithm
import os

Cx = 1
Cy = 1 / 200
Cz = 4
eps = 1e-6


def fairness_aware_dynamic_pricing(A: pmodel.PriceModel,p_min, p_max) -> (float, float):
    pp_max = p_max
    pp_min = p_min
    tau = np.power(A.T, 0.5) * np.log(A.T)
    tau_0 = tau
    tau_cg = 0.6
    kapa_cg = 0.6
    kapa = int(np.power(A.T, 0.1) * np.log(A.T))
    x = int(tau/kapa)
    y = (pp_max - pp_min) / (kapa + 1)
    t = 0
    while True:
        r_opt = 0
        p_opt = (pp_min + pp_max) / 2
        p_now = pp_min
        for i in range(0, kapa + 2):
            p_now = p_now + y
            r = 0
            for j in range(1, x + 1):
                d_1, d_2 = A.offer_price(p_now, p_now)
                t += 1
                if t >= A.T:
                    return p_opt, p_opt
                r += d_1 + d_2
            r = r * (p_now - A.c) / (x + 1)
            if r > r_opt:
                r_opt = r
                p_opt = p_now
        print(p_opt, tau, kapa, y)
        if y * y * np.sqrt(np.log2(A.T)) < tau_0 / A.T:
            break
        if p_opt < pp_min + eps:
            pp_max = pp_min + 2 * y / 3
        elif p_opt > pp_max - eps:
            pp_min = pp_max - y / 3
        else:
            pp_min = p_opt - y / 3
            pp_max = p_opt + 2 * y / 3
        tau = int(min(tau, np.power(tau, tau_cg) * np.log2(tau)))
        kapa = int(min(kapa, np.power(kapa, kapa_cg) * np.log2(kapa)))
        y = (pp_max - pp_min) / (kapa + 1)
        x = int(tau / kapa)
    for x in range(t + 1 , A.T + 1):
        A.offer_price(p_opt,p_opt)
    return p_opt, p_opt














