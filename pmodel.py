import random
import os
import sys

import numpy as np

K = 4.
C = 2

gamma = 1.

# dont change the p_min, p_max (:>)#
p_max = 4.
p_min = 0.5

lbda = 5.

# default = 0 for testing code, default + 1 for generate price function randomly using exp function
default = 1
eps = 1e-8

is_soft_constraint = False

class PriceModel:
    def __init__(self, t):
        self.regret = 0
        self.c = 0
        self.T = t
        self.t = 0

        self.r1d_0 = 0.1  # [0.1, 0.9]
        self.r1p_0 = 0.5  # [0.5, 1]
        self.r1eps = 0.5  # [0.5, 2] p_0
        self.p_1s = -1
        self.p_2s = -1
        self.r2d_0 = 0.1  # [0.1, 0.9]
        self.r2p_0 = 0.5  # [0.5, 1]
        self.r2eps = 0.5  # [0.5, 2] p_0
        self.r1p_max0 = 0
        self.r2p_max0 = 0
        self.r1p_max = 0
        self.r2p_max = 0

        self.d_1s = -1
        self.d_2s = -1
        self.rs = -1
        self.m1b = -1
        self.m2b = -1
        self.p1_best = -1
        self.p2_best = -1
        self.r_best = -1

    # re_gen is for regenerate the price function
    def re_gen(self, time, FixedSinglePrice: bool = False):
        self.T = time
        self.regret = 0
        self.t = 0
        self.p_1s = -1
        self.p_2s = -1
        self.d_1s = -1
        self.d_2s = -1
        self.rs = -1
        self.r_best = -1
        self.p1_best = -1
        self.p2_best = -1
        self.m1b = -1
        self.m2b = -1
        if default == 1:
            self.generate_price_function(FixedSinglePrice)

    # refresh is for refreshing the enumerator while keeping the same price function
    def refresh(self):
        self.regret = 0
        self.t = 0

    def generate_price_function(self, FixedSinglePrice: bool = False):
        # *** requirement:  d(p) = d_0 e^ [-eps (p / p0 - 1)]
        # p(d) = [1/ eps ln ( d_0 / d) + 1 ] p_0
        # d_0 - [0.1, 0.9]
        # c - [0, 1]
        # p_0 - [0.5, 1]
        # eps - [0.5, 2]p_0
        # max p = p_0 / eps + c

        if FixedSinglePrice:
            self.r1d_0 = self.r2d_0 = 0.5
            self.c = 0
            self.r1p_0 = 1
            self.r2p_0 = 1
            self.r1eps = 1
            self.r2eps = 0.5
        else:
            self.r1d_0 = random.random() * 0.8 + 0.1
            self.r2d_0 = random.random() * 0.8 + 0.1
            self.c = random.random()
            self.r1p_0 = random.random() * 0.5 + 0.5
            self.r2p_0 = random.random() * 0.5 + 0.5

            self.r1eps = self.r1p_0 / 2. / (random.random() + eps)
            self.r2eps = self.r2p_0 / 2. / (random.random() + eps)

            # self.r1eps = pow(2, random.random() * 14 - 7) * self.r1p_0
            # self.r2eps = pow(2, random.random() * 14 - 7) * self.r2p_0
            # original best price that can be offered to the customer
        self.r1p_max0 = self.r1p_0 / self.r1eps + self.c
        self.r2p_max0 = self.r2p_0 / self.r2eps + self.c

        #print(self.r1p_max0, self.r2p_max0)

        # restricted the best price in the [p_min, p_max]
        self.r1p_max = min(p_max, max(p_min, self.r1p_max0))
        self.r2p_max = min(p_max, max(p_min, self.r2p_max0))

        self.best_sol()

    def get_demand_1(self, p: float) -> float:
        if default == 0:
            return 0.5 - 0.25 * p
        if default == 1:
            return self.r1d_0 * np.exp(-self.r1eps * (p / self.r1p_0 - 1))

    def get_demand_2(self, p: float) -> float:
        if default == 0:
            return 1 - 0.25 * p
        if default == 1:
            return self.r2d_0 * np.exp(-self.r2eps * (p / self.r2p_0 - 1))

    def search_best_reward(self) -> float:
        if default == 0:
            return 15. / 16 + 63. / 64
        else:
            dist = self.r1p_max - self.r2p_max
            x = max(1., lbda) * dist / 2  #note that if lbda larger than 1 we should bound that to 1.
            l = p_min + np.abs(x)
            r = p_max - np.abs(x)
            while r - l > eps:
                p_1 = 2./3 * l + 1./3 * r
                p_2 = 1./3 * l + 2./3 * r
                t11 = self.get_demand_1(p_1 + x)
                t12 = self.get_demand_2(p_1 - x)
                t21 = self.get_demand_1(p_2 + x)
                t22 = self.get_demand_2(p_2 - x)
                r_1 = t11 * (p_1 + x - self.c) + t12 * (p_1 - x - self.c)
                r_2 = t21 * (p_2 + x - self.c) + t22 * (p_2 - x - self.c)
                if r_1 < r_2:
                    l = p_1
                else:
                    r = p_2
                self.p1_best = p_1 + x
                self.p2_best = p_1 - x
                self.r_best = r_1
                self.m1b = self.get_demand_1(self.p1_best)
                self.m2b = self.get_demand_2(self.p2_best)
            return self.r_best

    def best_sol(self) -> (float, float):
        if self.p1_best == -1:
            p = self.search_best_reward()
        print("Best Solution: ", self.r1p_max, ' ', self.r2p_max, ' ', self.p1_best, ' ', self.p2_best, ' ', self.r_best)
        return self.p1_best, self.p2_best

    def regret_cal(self, p_1, p_2) -> (float, float, float):    # d_1, d_2, regret

        if self.p1_best == -1:
            best_R = self.search_best_reward()
        else:
            best_R = self.r_best

        if self.p_1s == p_1 and self.p_2s == p_2:
            return self.d_1s, self.d_2s, self.rs
        else:
            self.p_1s = p_1
            self.p_2s = p_2
            self.d_1s = self.get_demand_1(p_1)
            self.d_2s = self.get_demand_2(p_2)

            R = self.d_1s * (p_1 - self.c) + self.d_2s * (p_2 - self.c)
            if is_soft_constraint:
                # m1, m2 = self.get_m(p_1,p_2)
                # m1b, m2b = self.get_m(self.p1_best, self.p2_best)
                m1 = self.d_1s
                m2 = self.d_2s
                self.rs = max(best_R - R, 0) + gamma * max(np.abs(m1 - m2) - lbda * np.abs(self.m1b - self.m2b), 0)
                return self.d_1s, self.d_2s, self.rs
            else:
                self.rs = max(best_R - R, 0)
            return self.d_1s, self.d_2s, self.rs

    #  def regret_cal_soft(self, p_1, p_2) -> (float, float, float):    # d_1, d_2, regret
    #      if self.p1_best == -1:
    #          best_R = self.search_best_reward()
    #      else:
    #          best_R = self.r_best
    #      d_1 = self.get_demand_1(p_1)
    #      d_2 = self.get_demand_2(p_2)
    #      m1, m2 = self.get_m(p_1,p_2)
    #
    #      m1b, m2b = self.get_m(self.p1_best, self.p2_best)

    #      R = d_1 * (p_1 - self.c) + d_2 * (p_2 - self.c) - gamma * max(np.abs(m1 - m2) - lbda * np.abs(m1b - m2b), 0)
    #      return d_1, d_2, best_R - R

    def offer_price(self, p_1: float, p_2: float) -> (float, float):
        self.t += 1
        d_1, d_2, regret = self.regret_cal(p_1, p_2)

        if regret < 0:
            print("error, regret negative:", p_1, p_2, regret)

        self.regret += regret
        return float(random.random() < d_1), float(random.random() < d_2)

    def offer_price_soft(self, p_1: float, p_2: float) -> (float, float, float, float):
        self.t += 1
        d_1, d_2, regret = self.regret_cal(p_1, p_2)
        if regret < 0:
            print("error, regret negative:", p_1, p_2, regret)

        self.regret += regret

        # m_1, m_2 = self.get_m(p_1, p_2)

        d1 = float(random.random() < d_1)
        d2 = float(random.random() < d_2)
        return d1, d2, d1, d2

#  def get_m(self, p_1: float, p_2: float) -> (float, float):
#     return p_1, p_2

    def set_time(self, time):
        self.T = time

    def get_time(self):
        print("Current selling periods:", self.t)

    def get_regret(self):
        print("Current Regret: ", self.regret)
        return self.regret
