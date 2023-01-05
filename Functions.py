import numpy as np
from pulp import LpMaximize, LpProblem, lpDot, LpVariable
from copy import deepcopy
import sys

EPS = sys.float_info.epsilon


def DLH(G):
    # D(x) = {y in X | yPx}
    # L(x) = {y in X | xPy}
    # H(x) = {y in X | (not xPy) and (not yPx) and (x != y)}
    D = []
    L = []
    H = []
    for i in range(len(G)):
        D.append(set(j for j in range(len(G)) if G[i][j] == -1))
        L.append(set(j for j in range(len(G)) if G[i][j] == 1))
        H.append(set(j for j in range(len(G)) if G[i][j] == 0 and i != j))
    return {"D": D, "L": L, "H": H}


def CO_2(G):
    # s2(x) = |L(x)|
    L = DLH(G)['L']
    s2_list = list(map(len, L))
    return s2_list


def CO_3(G):
    # s3(x) = |X| - |D(x)|
    card_X = len(G)
    D = DLH(G)['D']
    s3_list = [card_X - len(D_x) for D_x in D]
    return s3_list


def UC_M(G):
    # x C_M y <=> xPy and L(y) < L(x)
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (L[j] < L[i]):
                covered.add(j)
    return all_alt_set - covered


def UC_F(G):
    # x C_F y <=> xPy and D(x) < D(y)
    all_alt_set = set(i for i in range(len(G)))
    D = DLH(G)["D"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (D[i] < D[j]):
                covered.add(j)
    return all_alt_set - covered


def UC_McK(G):
    # x C_McK y <=> xPy and D(x) < D(y) and L(y) < L(x)
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    D = DLH(G)["D"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (D[i] < D[j]) and (L[j] < L[i]):
                covered.add(j)
    return all_alt_set - covered


def UC_D(G):
    # x C_D y <=> H(y) union L(y) < L(x)
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    H = DLH(G)["H"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if H[j].union(L[j]) < L[i]:
                covered.add(j)
    return all_alt_set - covered


def ES(G):
    # x in ES <=> exists y in UC_F: xPy or x in UC_F
    res = set()
    D = DLH(G)
    UC_F_set = UC_F(G)
    for y in UC_F_set:
        res = res.union(D[y])
    return res.union(UC_F_set)


def E(G):
    # E = union of supports of all maximal lotteries
    G = np.array(G)
    zeros = [0] * (len(G[0]) * 2 + 1)
    list_add = []
    for i in range(1, len(G[0]) * 2 + 1):
        temp = deepcopy(zeros)
        temp[i] = 1  # 1 pivot
        list_add.append(temp)
    z_0 = [0 for i in range(len(G) + 1)]
    zeros.pop(0)
    z = [*z_0, -1000, *zeros]
    z[0] = 1
    zeros.append(0)
    A = []
    A_1 = [0, *[1 for _ in range(len(G))], 1, *[0 for _ in range(len(G) * 2)]]
    A.append(A_1)
    i = 0
    for row in np.vstack((G, G)):
        if i <= len(G) - 1:
            A_i = [0, *row, *(list_add[i])]
        else:
            temp = deepcopy(row)
            temp[i - len(G)] -= 1
            A_i = [1, *temp, *(list_add[i])]
        A.append(A_i)
        i += 1
    b = [1, *[0 for _ in range(len(A) - 1)]]
    # Построение модели
    E = LpProblem(name="E", sense=LpMaximize)
    # Переменные задачи:
    all_ind = [0]
    for el in list_ind:
        all_ind.append(el + 1)
    to_add_ind = "*"
    for i in range(len(A[0]) - len(list_ind) - 1):  #######
        all_ind.append(to_add_ind)
        to_add_ind += "*"
    p = [LpVariable(name='p' + str(i), lowBound=0) for i in all_ind]  ######
    # Добавляем в модель ограничения:
    for k in [i for i in range(len(G[0]) * 2 + 1)]:
        E += (lpDot(A[k], p) == b[k]), '(' + str(k + 1) + ')'
    # Добавляем целевую функцию:
    E += lpDot(z, p)
    # Смотрим, что получилось:
    res = []
    for v in E.variables():
        if v.varValue > EPS and "*" not in v.name and v.name != 'p0':  ######
            res.append(int(v.name.replace('p', '')) - 1)
    return res


def MC_McK():
    pass


def MC_D():
    pass
