import sys
import numpy as np
from pulp import LpMaximize, LpProblem, lpDot, LpVariable

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


def G_reduction(G, el_to_del):
    G = np.delete(G, el_to_del, 0)
    return np.delete(G, el_to_del, 1)


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
    D = DLH(G)["D"]
    UC_F_set = UC_F(G)
    for y in UC_F_set:
        res = res.union(D[y])
    return res.union(UC_F_set)


def E(G):
    # E = объединение носителей всех максимальных лотерей
    list_ind = [i for i in range(len(G))]
    # Вектор коэффициентов целевой функции
    z = [1, *[0 for _ in range(len(G))], -1000, *[0] * (len(G[0]) * 2)]
    # Матрица коэффициентов левой части
    A = []
    A_0 = [0, *[1 for _ in range(len(G))], 1, *[0 for _ in range(len(G) * 2)]]
    A.append(A_0)
    i = 0
    for row in np.vstack((G, G)):
        to_add = [0] * (len(G[0]) * 2 + 1)
        to_add[i + 1] = 1
        if i <= len(G) - 1:
            A_i = [0, *row, *to_add]
        else:
            row[i - len(G)] -= 1
            A_i = [1, *row, *to_add]
        A.append(A_i)
        i += 1
    # Вектор коэффициентов правой части
    b = [1, *[0 for _ in range(len(A) - 1)]]
    # Построение модели
    E = LpProblem(name="E", sense=LpMaximize)
    # Переменные задачи
    all_ind = list(map(str, list_ind)) + [str(list_ind[-1] + 1)]
    for i in range(len(A[0]) - len(list_ind) - 1):
        all_ind.append("*" * (i + 1))
    p = [LpVariable(name='p' + i, lowBound=0) for i in all_ind]
    # Ограничения
    for k in [i for i in range(len(G[0]) * 2 + 1)]:
        E += (lpDot(A[k], p) == b[k]), '(' + str(k + 1) + ')'
    # Целевая функция
    E += lpDot(z, p)
    # Решение задачи
    E.solve()
    res = set()
    for v in E.variables():
        if v.varValue > EPS and "*" not in v.name and v.name != 'p0':
            res.add(int(v.name.replace('p', '')) - 1)
    return res


def MC_McK(G):
    B = E(G)
    all_alt = set([x for x in range(len(G))])
    while True:
        temp_res = set()
        for el in all_alt - B:
            el_to_del = list(all_alt - B.union({el}))
            UC_McK_set = UC_McK(G_reduction(G, el_to_del))
            el_to_stay = list(B.union({el}))
            el_to_stay.sort()
            UC_new = set()
            for ind in UC_McK_set:
                UC_new.add(el_to_stay[ind])
            if UC_new.intersection({el}) != set():
                temp_res.add(el)
        if len(temp_res) == 0:
            return B
        else:
            el_to_del = list(all_alt - temp_res)
            el_to_stay = list(temp_res)
            el_to_stay.sort()
            A_new = E(G_reduction(G, el_to_del))
            to_add = set()
            for ind in A_new:
                to_add.add(el_to_stay[ind])
            B = B.union(to_add)


def MC_D(G):
    B = E(G)
    all_alt = set([x for x in range(len(G))])
    while True:
        temp_res = set()
        for el in all_alt - B:
            el_to_del = list(all_alt - B.union({el}))
            UC_D_set = UC_D(G_reduction(G, el_to_del))
            el_to_stay = list(B.union({el}))
            el_to_stay.sort()
            UC_new = set()
            for ind in UC_D_set:
                UC_new.add(el_to_stay[ind])
            if UC_new.intersection({el}) != set():
                temp_res.add(el)
        if len(temp_res) == 0:
            return B
        else:
            el_to_del = list(all_alt - temp_res)
            el_to_stay = list(temp_res)
            el_to_stay.sort()
            A_new = E(G_reduction(G, el_to_del))
            to_add = set()
            for ind in A_new:
                to_add.add(el_to_stay[ind])
            B = B.union(to_add)


def sorting(G, func):
    rank_all = []
    N = len(G)
    if func in [CO_2, CO_3]:
        index_all = [x for x in range(len(G))]
        score = func(G)
        dict_temp = dict(zip(index_all, score))
        keys_sorted = sorted(dict_temp, key=dict_temp.get, reverse=True)
        value_max = dict_temp[keys_sorted[0]]
        rank_temp = []
        while value_max != -1:
            for key in keys_sorted:
                value_now = dict_temp[key]
                if value_now == value_max:
                    dict_temp[key] = -1
                    rank_temp.append(key)
            rank_all.append(rank_temp)
            rank_temp = []
            keys_sorted = sorted(dict_temp, key=dict_temp.get, reverse=True)
            value_max = dict_temp[keys_sorted[0]]
    else:
        index_left = [x for x in range(len(G))]
        while True:
            if len(G) == 0:
                break
            rank_temp = []
            res_temp = func(G)
            for el_ind in res_temp:
                rank_temp.append(index_left[el_ind])
            index_left = [index_left[i] for i in range(len(index_left)) if i not in res_temp]
            G = G_reduction(G, list(res_temp))
            rank_all.append(rank_temp)
    place = 1
    ranking = [0] * N
    for place_list in rank_all:
        for i in place_list:
            ranking[i] = place
        place += 1
    return ranking


def init_ranking(df):
    rankings = list(df.columns)
    for ranking in rankings:
        res = np.zeros(len(df))
        place = 1
        maximum = df[ranking].max()
        while maximum != -100:
            ind_list = np.where(df[ranking] == maximum)
            for ind in ind_list:
                df[ranking][ind] = -100
                res[ind] = place
            place += 1
            maximum = df[ranking].max()
        df[ranking] = res
    df = df.astype("int")
    return df


def M_T_generator(df):
    values = df.values
    M = np.zeros((values.shape[0], values.shape[0]))
    T = np.zeros((values.shape[0], values.shape[0]))
    num = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[0]):
            counter = 0
            for k in range(1, values.shape[1]):  # идем по ранжированиям
                if values[i][k] < values[j][k]:
                    counter += 1
                elif values[i][k] > values[j][k]:
                    counter -= 1
            if np.sign(counter) == 1:
                M[i][j] = 1
            elif np.sign(counter) == 0 and i != j:
                T[i][j] = 1
                num += 1
    return M.astype(np.int64), T.astype(np.int64), num // 2


def game_creator(M):
    return M - M.T
