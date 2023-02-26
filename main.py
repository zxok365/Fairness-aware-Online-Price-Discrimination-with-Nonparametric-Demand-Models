import random
import pmodel
import algorithm
import algorithm_new
import algorithm_soft_constraint
import numpy as np
import idle_algorithm
import csv
import baseline_zizhuo
eps = 1e-6
# C = 100000

NN = 100
re_avg1 = 0
re_avg2 = 0
re_avg3 = 0
Fixed_Single_Price = True

for t in range(1, 11):
    C = t * 100000
    A = pmodel.PriceModel(C)
    if not pmodel.is_soft_constraint:
        with open("Try" + str(C) + ".csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["lambda=" + str(pmodel.lbda), "T=" + str(C), "Single_Price=" + str(Fixed_Single_Price),
                             "Alg.Cx=" + str(algorithm_new.Cx), "Alg.Cy=" + str(algorithm_new.Cy),
                             "Alg.Cz=" + str(algorithm_new.Cz), "soft_constraint=" + str(pmodel.is_soft_constraint)])

        for tt in range(1, NN + 1):
            print("Round", tt, ":")
            A.re_gen(C, Fixed_Single_Price)
            print("Our Algorithm gives:")
            # p_11, p_12 = algorithm.fairness_aware_dynamic_pricing(A)

            p_11, p_12 = algorithm_new.fairness_aware_dynamic_pricing(A)
            print(A.p1_best, A.p2_best, A.regret_cal(A.p1_best, A.p2_best))
            print(p_11, ' ', p_12, ' ', A.regret_cal(p_11, p_12))
            print("price fairness condition:", np.abs(A.p1_best - A.p2_best), np.abs(p_11 - p_12))
            re1 = A.get_regret()
            A.get_time()

            A.refresh()
            print("")
            print("***Idle Algorithm gives:")
            p_21, p_22 = idle_algorithm.fairness_aware_dynamic_pricing(A, pmodel.p_min, pmodel.p_max)
            print("***", p_21, ' ', p_22, ' ', A.regret_cal(p_21, p_22))
            re2 = A.get_regret()
            A.get_time()

            A.refresh()
            p_31, p_32 = baseline_zizhuo.fairness_aware_dynamic_pricing(A, pmodel.p_min, pmodel.p_max)
            print("***", p_31, ' ', p_32, ' ', A.regret_cal(p_31, p_32))
            print("")
            re3 = A.get_regret()
            A.get_time()


            print(re1, re2, re3)
            re_avg1 += re1
            re_avg2 += re2
            re_avg3 += re3
            with open("Try" + str(C) + ".csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([tt, np.abs(A.p1_best-A.p2_best), A.p1_best, A.p2_best, re1, re2, re3, p_11, p_12, p_21, p_22, p_31, p_32, np.abs(A.p1_best - A.p2_best), np.abs(p_11 - p_12)])
        print(re_avg1 / NN, re_avg2 / NN, re_avg3 / NN)
    else:
        with open("Soft" + str(C) + ".csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["lambda=" + str(pmodel.lbda), "T=" + str(C), "Single_Price=" + str(Fixed_Single_Price),
                             "Alg.Cx=" + str(algorithm_new.Cx), "Alg.Cy=" + str(algorithm_new.Cy),
                             "Alg.Cz=" + str(algorithm_new.Cz), "soft_constraint=" + str(pmodel.is_soft_constraint)])


        for tt in range(1, NN + 1):
            print("Round", tt, ":")
            A.re_gen(C, Fixed_Single_Price)
            print("Our Algorithm gives:")
            # p_11, p_12 = algorithm.fairness_aware_dynamic_pricing(A)

    #       p_11, p_12 = algorithm_new.fairness_aware_dynamic_pricing(A)
    #       print(A.p1_best, A.p2_best, A.regret_cal(A.p1_best, A.p2_best))
    #        print(p_11, ' ', p_12, ' ', A.regret_cal(p_11, p_12))
    #        print("price fairness condition:", np.abs(A.p1_best - A.p2_best), np.abs(p_11 - p_12))
    #        re1 = A.get_regret()
    #        A.get_time()
    #        print("")
    #        print("***Idle Algorithm gives:")

            p_11, p_12 = algorithm_soft_constraint.fairness_aware_dynamic_pricing(A)
            print(A.p1_best, A.p2_best, A.regret_cal(A.p1_best, A.p2_best))
            print(p_11, ' ', p_12, ' ', A.regret_cal(p_11, p_12))
            print("price fairness condition:", np.abs(A.p1_best - A.p2_best), np.abs(p_11 - p_12))
            re1 = A.get_regret()
            A.get_time()

            A.refresh()
            p_21, p_22 = idle_algorithm.fairness_aware_dynamic_pricing(A, pmodel.p_min, pmodel.p_max)
            print("***", p_21, ' ', p_22, ' ', A.regret_cal(p_21, p_22))
            print("")
            re2 = A.get_regret()
            A.get_time()

            A.refresh()
            p_31, p_32 = baseline_zizhuo.fairness_aware_dynamic_pricing(A,pmodel.p_min,pmodel.p_max)
            print(A.p1_best, A.p2_best, A.regret_cal(A.p1_best, A.p2_best))
            print(p_31, ' ', p_32, ' ', A.regret_cal(p_31, p_32))
            print("price fairness condition:", np.abs(A.p1_best - A.p2_best), np.abs(p_31 - p_32))
            re3 = A.get_regret()
            A.get_time()


            re_avg1 += re1
            re_avg2 += re2
            re_avg3 += re3
            print(re1, re2, re3)
            with open("Soft" + str(C) + ".csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([tt, np.abs(A.p1_best - A.p2_best), A.p1_best, A.p2_best, re1, re2, re3, p_11, p_12, p_21, p_22, p_31, p_32,
                                 np.abs(A.p1_best - A.p2_best), np.abs(p_11 - p_12), np.abs(p_31 - p_32)])

        print(re_avg1 / NN, re_avg2 / NN, re_avg3 / NN)