import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, lpDot, LpVariable, value


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


def ES(G):
    # x in ES <=> exists y in UC_F: xPy or x in UC_F
    res = set()
    D = DLH(G)
    UC_F_set = UC_F(G)
    for y in UC_F_set:
        res = res.union(D[y])
    return res.union(UC_F_set)


def E(G):
    # E = union of supports of all Nash equilibrium
    pass


def MC_McK():
    pass


def MC_D():
    pass
