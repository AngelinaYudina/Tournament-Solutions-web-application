import sys
from typing import Callable, Union
import numpy as np
from pandas.io.excel._xlsxwriter import ExcelWriter
import pandas as pd
from pulp import LpMaximize, LpProblem, lpDot, LpVariable

# Машинный эпсилон для числа с плавающей точкой
EPS = sys.float_info.epsilon


def init_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает таблицу ранжирований (без пропусков).
    :param df: таблица рейтингов
    :return: таблица ранжирований (без пропусков)
    """
    rankings = list(df.columns)
    for ranking in rankings:
        res = np.zeros(len(df))
        place = 1
        maximum = df[ranking].max()
        while maximum != -100:
            ind_list = np.where(np.abs(df[ranking] - maximum) <= EPS)
            for ind in ind_list:
                df[ranking][ind] = -100
                res[ind] = place
            place += 1
            maximum = df[ranking].max()
        df[ranking] = res
    df = df.astype("int")
    return df


def M_T_generator(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Создает матрицу строго строгого мажоритарного отношения и матрицу отношения равенства голосов.
    Рассчитает количество пар альтарнатив, находящихся в отношении эквивалентности.
    :param df: таблица ранжирований (без пропусков)
    :return: матрица строгого мажоритарного отношения, матрица отношения равенства голосов, количество ничьих
    """
    values = df.values
    M = np.zeros((values.shape[0], values.shape[0]))
    T = np.zeros((values.shape[0], values.shape[0]))
    num = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[0]):
            counter = 0
            for k in range(0, values.shape[1]):  # проход по ранжированиям
                if values[i][k] < values[j][k]:
                    counter += 1
                elif values[i][k] > values[j][k]:
                    counter -= 1
            if np.sign(counter) == 1:
                M[i][j] = 1
            elif np.sign(counter) == 0 and i != j:
                T[i][j] = 1
                T[j][i] = 1
                num += 1
    return M.astype(np.int64), T.astype(np.int64), num // 2


def game_creator(M: np.ndarray) -> np.ndarray:
    """
    Создает матрицу турнирной игры.
    :param M: матрица строгого мажоритарного отношения.
    :return: матрица турнирной игры
    """
    return M - M.T


def G_reduction(G: np.ndarray, el_to_del: list[int]) -> np.ndarray:
    """
    Удаляет из предъявления все альтернативы с индексом из заданного списка. Создает на оставшихся альтернативах
    новый турнир.
    :param G: матрица турнирной игры
    :param el_to_del: список индексов (название альтернативы)
    :return: матрица турнирной игры (для нового турнира)
    """
    G = np.delete(G, el_to_del, 0)
    return np.delete(G, el_to_del, 1)


def DLH(G: np.ndarray) -> dict[str, list[set[int]]]:
    """
    Вычисляет верхний и нижний срезы и горизонт каждой альтернативы.

    Верхний срез альтернативы x: D(x) = {y in X | yPx}

    Нижный срез альтернативы x: L(x) = {y in X | xPy}

    Горизонт альтернативы x: H(x) = {y in X | (not xPy) and (not yPx) and (x != y)}
    :param G: матрица турнирной игры
    :return: верхний и нижний срезы и горизонт каждой альтернативы предъявления в виде словаря с ключами "D", "L", "H"
    """
    D = []
    L = []
    H = []
    for i in range(len(G)):
        D.append(set(j for j in range(len(G)) if G[i][j] == -1))
        L.append(set(j for j in range(len(G)) if G[i][j] == 1))
        H.append(set(j for j in range(len(G)) if G[i][j] == 0 and i != j))
    return {"D": D, "L": L, "H": H}


def CO_2(G: np.ndarray) -> list[int]:
    """
    Вычисляет оценки Коупланда s2 (2 версия) для всех альтернатив предъявления.

    s2(x) = |L(x)|
    :param G: матрица турнирной игры
    :return: список оценок Коупланда s2 (2 версия)
    """
    L = DLH(G)['L']
    s2_list = list(map(len, L))
    return s2_list


def CO_3(G: np.ndarray) -> list[int]:
    """
    Вычисляет оценки Коупланда s3 (3 версия) для всех альтернатив предъявления.

    s3(x) = |X| - |D(x)|
    :param G: матрица турнирной игры
    :return: список оценок Коупланда s3 (3 версия)
    """
    card_X = len(G)
    D = DLH(G)['D']
    s3_list = [card_X - len(D_x) for D_x in D]
    return s3_list


def UC_M(G: np.ndarray) -> set[int]:
    """
    Вычисляет непокрытое по Миллеру множество UC_M.

    Версия Миллера отношения покрытия:

    x C_M y <=> xPy and L(y) < L(x).

    Альтернатива называется непокрытой по Миллеру тогда и только тогда, когда в предъявлении нет ни одной альтернативы,
    покрывающей её по Миллеру. Непокрытое по Миллеру множество UC_M есть множество всех непокрытых по версии Миллера
    отношения покрытия альтернатив.
    :param G: матрица турнирной игры
    :return: непокрытое по Миллеру множество UC_M
    """
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (L[j] < L[i]):
                covered.add(j)
    return all_alt_set - covered


def UC_F(G: np.ndarray) -> set[int]:
    """
    Вычисляет непокрытое по Фишберну множество UC_F.

    Версия Фишберна отношения покрытия:

    x C_F y <=> xPy and D(x) < D(y).

    Альтернатива называется непокрытой по Фишберну тогда и только тогда, когда в предъявлении нет ни одной альтернативы,
    покрывающей её по Фишберну. Непокрытое по Фишберну множество UC_F есть множество всех непокрытых по версии Фишберна
    отношения покрытия альтернатив.
    :param G: матрица турнирной игры
    :return: непокрытое по Фишберну множество UC_F
    """
    all_alt_set = set(i for i in range(len(G)))
    D = DLH(G)["D"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (D[i] < D[j]):
                covered.add(j)
    return all_alt_set - covered


def UC_McK(G: np.ndarray) -> set[int]:
    """
    Вычисляет непокрытое по МакКельви множество UC_McK.

    Версия МакКельви отношения покрытия:

    x C_McK y <=> xPy and D(x) < D(y) and L(y) < L(x).

    Альтернатива называется непокрытой по МакКельви тогда и только тогда, когда в предъявлении нет ни одной
    альтернативы, покрывающей её по МакКельви. Непокрытое по МакКельви множество UC_McK есть множество всех непокрытых
    по версии МакКельви отношения покрытия альтернатив.
    :param G: матрица турнирной игры
    :return: непокрытое по МакКельви множество UC_McK
    """
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    D = DLH(G)["D"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and (D[i] < D[j]) and (L[j] < L[i]):
                covered.add(j)
    return all_alt_set - covered


def UC_D(G: np.ndarray) -> set[int]:
    """
    Вычисляет непокрытое по Дуггану множество UC_D.

    Версия Дуггана отношения покрытия:

    x C_D y <=> H(y) union L(y) < L(x).

    Альтернатива называется непокрытой по Дуггану тогда и только тогда, когда в предъявлении нет ни одной альтернативы,
    покрывающей её по Дуггану. Непокрытое по Дуггану множество UC_D есть множество всех непокрытых по версии Дуггана
    отношения покрытия альтернатив.
    :param G: матрица турнирной игры
    :return: непокрытое по Дуггану множество UC_D
    """
    all_alt_set = set(i for i in range(len(G)))
    L = DLH(G)["L"]
    H = DLH(G)["H"]
    covered = set()
    for i in range(len(G)):
        for j in range(len(G)):
            if H[j].union(L[j]) < L[i]:
                covered.add(j)
    return all_alt_set - covered


def ES(G: np.ndarray) -> set[int]:
    """
    Вычисляет объединение минимальных внешне устойчивых множеств ES.

    x in ES <=> exists y in UC_F: xPy or x in UC_F

    Алгоритм вычисления взят из статьи:

    Subochev, A. Dominating, weakly stable, and uncovered sets: Properties and generalizations. / A. Subochev //
    Automation and Remote Control. – 2010. – Vol. 71(1). – p. 116-127.
    :param G: матрица турнирной игры
    :return: объединение минимальных внешне устойчивых множеств ES
    """
    res = set()
    D = DLH(G)["D"]
    UC_F_set = UC_F(G)
    for y in UC_F_set:
        res = res.union(D[y])
    return res.union(UC_F_set)


def E(G: np.ndarray) -> set[int]:
    """
    Вычисляет существенное множество E, определяемое как объединение носителей всех максимальных лотерей слабого
    турнира.

    Алгоритм вычисления взят из статьи:

    Brandt, F. Computing the minimal covering set. / F. Brandt, F. Fischer // Mathematical Social Sciences. – 2008. –
    Vol. 56(2). – p. 254–268.
    :param G: матрица турнирной игры
    :return: существенное множество E
    """
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


def MC_McK(G: np.ndarray) -> set[int]:
    """
    Вычисляет минимальное покрывающее по МакКельви множество MC_McK.

    В слабом турнире покрывающим по МакКельви множеством Y (Y <= X) называют множество, удовлетворяющее следующим двум
    условиям:

    1. UC_McK(Y) = Y
    2. forall x in (X setminus Y), x not in UC_McK(Y cup {x}).

    Минимальное покрывающее по МакКельви множество определяется как единственное минимальное по вложению покрывающее по
    МакКельви множество.

    Алгоритм вычисления взят из статьи:

    Brandt, F. Computing the minimal covering set. / F. Brandt, F. Fischer // Mathematical Social Sciences. – 2008. –
    Vol. 56(2). – p. 254–268.
    :param G: матрица турнирной игры
    :return: минимальное покрывающее по МакКельви множество MC_McK
    """
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


def MC_D(G: np.ndarray) -> set[int]:
    """
    Вычисляет минимальное покрывающее по Дуггану множество MC_D.

    В слабом турнире покрывающим по Дуггану множеством Y (Y <= X) называют множество, удовлетворяющее следующим двум
    условиям:

    1. UC_D(Y) = Y
    2. forall x in (X setminus Y), x not in UC_D(Y cup {x}).

    Минимальное покрывающее по Дуггану множество определяется как единственное минимальное по вложению покрывающее по
    Дуггану множество.

    Алгоритм вычисления MC_D аналогичен алгоритму вычисления MC_McK (с точностью до переопределения версии
    отношения покрытия).
    :param G: матрица турнирной игры
    :return: минимальное покрывающее по Дуггану множество MC_D
    """
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


def sorting(G: np.ndarray, func: Callable[[np.ndarray], Union[set[int], list[int]]]) -> list[int]:
    """
    Производит сортировку альтернатив из предъявления по правилу, заданному функцией func.
    :param G: матрица турнирной игры
    :param func: турнирное решение
    :return: ранжировка, полученная путем сортировки альтернатив заданным турнирным решением
    """
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


def custom_format(writer: ExcelWriter, mode: str, rotation: int = 0) -> ExcelWriter:
    """
    Создает шаблон для форматирования чисел и заголовков.
    :param writer: объект, записывающий данные на лист Excel (с исходным форматированнием)
    :param mode: тип форматируемых значений. "base" - базовый, "header_col" - заголовок (для столбцов),
    "header_row" - заголовок (для строк)
    :param rotation: угол поворота текста (в градусах) (значение по умолчанию: 0)
    :return: объект, записывающий данные на лист Excel (с измененным форматированием)
    """
    style_dict = {
        "font_name": "Times New Roman",
        "font_size": 12,
        "bold": False,
        "align": "center",
        "valign": "center"
    }
    match mode:
        case "base":
            return writer.book.add_format(style_dict)
        case "header_col":
            style_dict["rotation"] = rotation
            return writer.book.add_format(style_dict)
        case "header_row":
            style_dict["align"] = "left"
            style_dict["valign"] = "left"
            return writer.book.add_format(style_dict)
