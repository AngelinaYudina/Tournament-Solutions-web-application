import sys
import os
import streamlit as st
import streamlit_ext as ste   # Позволяет скачивать файлы без перезагрузки страницы
import pandas as pd
import pandas.io.formats.excel
from Functions import init_ranking, M_T_generator, game_creator, sorting, CO_2, CO_3, UC_M, ES, E, MC_McK, MC_D, \
    custom_format

st.set_page_config(
    page_title="Bibliometric IS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "mailto:avyudina_2@edu.hse.ru",
        "About": """
        Автор: Ангелина Юдина ([GitHub](https://github.com/AngelinaYudina), [cайт](https://www.hse.ru/staff/ayudina))  
        Стажер-исследователь в [Международном центре анализа и выбора решений НИУ ВШЭ](https://www.hse.ru/DeCAn/)
        """
    }
)

# Машинный эпсилон для числа с плавающей точкой
EPS = sys.float_info.epsilon

st.title("Библиометрическая информационная система оценки научных журналов")
st.write("Пожалуйста, загрузите свой файл. Если у вас нет готового файла, вы можете скачать пример данных и "
         "работать с ним.")
st.warning(""" 
Убедитесь, что во всех рейтингах наибольшее значение соответствует наилучшей альтернативе!  
При загрузке csv файла используйте запятую (,) в качестве разделителя! 
""")
# Выгрузка образца данных
with open("sample_data.xlsx", "rb") as file:
    ste.download_button(
        label="Скачать пример данных",
        data=file,
        file_name="sample_data.xlsx"
    )
# Загрузка файла
if "file" not in st.session_state:
    st.session_state["file"] = None
file = st.file_uploader("Загрузите файл", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
st.session_state["file"] = file
# Чтение файла
if file is not None:
    st.success("Файл успешно загружен!")
    try:
        df = pd.read_csv(file, delimiter=",")
    except:
        df = pd.read_excel(file)
    # Создание матрицы турнирной игры G на основе данных из df
    df = df.set_index(df.columns[0])
    rankings_all = init_ranking(df)
    M, T, counter = M_T_generator(rankings_all)
    G = game_creator(M)
    st.write(f"Количество журналов = {df.shape[0]}. Количество ничьих = {counter}. Исходные ранжирования: "
             f"{list(df.columns)}")
    # Построение ранжирований сортировкой, основанной на различных турнирных решениях
    tournament_solutions = [CO_2, CO_3, UC_M, ES, E, MC_McK, MC_D]
    sol_names = ["Copeland rule (2 v.)", "Copeland rule (3 v.)", "sorting by UC_M", "sorting by ES", "sorting by E",
                 "sorting by MC_McK", "sorting by MC_D"]
    for i in range(len(tournament_solutions)):
        ranking = sorting(G, tournament_solutions[i])
        rankings_all[sol_names[i]] = ranking
    st.write("Ранжирования журналов")
    st.dataframe(rankings_all)
    # Расчет коэффициента ранговой корреляции τ_b Кендалла
    corr = rankings_all.corr(method='kendall')
    ind = corr.columns
    rounded_corr = corr.round(3)   # Таблицу с округленными значениями
    st.write("Значения коэффициента τ_b Кендалла")
    st.dataframe(rounded_corr)
    # Построение матрицы строгого мажоритарного отношения на основе таблицы со значениями коэффициента τ_b
    matrix = [[0 for _ in range(len(ind))] for _ in range(len(ind))]
    for i in range(len(ind)):
        for j in range(len(ind)):
            val_i = 0
            val_j = 0
            if i < j:
                for k in range(len(df.columns)):
                    if corr.iloc[k, i] - corr.iloc[k, j] > EPS:
                        val_i += 1
                    elif corr.iloc[k, i] - corr.iloc[k, j] < - EPS:
                        val_j += 1
                if val_i > val_j:
                    matrix[i][j] = 1
                elif val_i < val_j:
                    matrix[j][i] = 1
    matrix_pd = pd.DataFrame(matrix, index=ind, columns=ind)
    res = pd.DataFrame(matrix, columns=ind, index=ind)
    sum_col = []
    for row in matrix:
        sum_row = sum(row)
        sum_col.append(sum_row)
    res["Copeland score s2"] = sum_col
    st.write("Матрица строгого мажоритарного отношения (τ_b)")
    st.dataframe(res)
    # Сравнение методов построения ранжирований
    pos = [x + 1 for x in range(len(ind))]
    res_sorted = res.sort_values(by="Copeland score s2", ascending=False)
    ind_ranked = res_sorted.index
    final_ranking = pd.DataFrame(ind_ranked, columns=["Метод"], index=pos)
    rank = []
    counter = 1
    value_prev = res_sorted["Copeland score s2"].values.max()
    for value in res_sorted["Copeland score s2"]:
        if value < value_prev:
            counter += 1
        rank.append(counter)
        value_prev = value
    final_ranking["Ранг"] = rank
    st.write("Ранжирование ранжирований")
    final_ranking = final_ranking.set_index("Ранг")
    st.dataframe(final_ranking)
    # Подготовка файла с результатом работы программы
    st.write("Все результаты в одном файле:")
    pandas.io.formats.excel.ExcelFormatter.header_style = None   # Разрешение на форматирование индексов
    with pd.ExcelWriter("Results.xlsx", engine="xlsxwriter") as writer:
        rankings_all.to_excel(writer, sheet_name="Rankings")
        pd.DataFrame(M).to_excel(writer, sheet_name="M", header=False, index=False)
        pd.DataFrame(T).to_excel(writer, sheet_name="T", header=False, index=False)
        rounded_corr.to_excel(writer, sheet_name="τ_b")
        res.to_excel(writer, sheet_name="M (τ_b)")
        final_ranking.to_excel(writer, sheet_name="Ranking of rankings (τ_b)")
        # Настройка стиля
        sheet_Rankings = writer.sheets["Rankings"]
        sheet_M = writer.sheets["M"]
        sheet_T = writer.sheets["T"]
        sheet_t_b = writer.sheets["τ_b"]
        sheet_M_t_b = writer.sheets["M (τ_b)"]
        sheet_final_res = writer.sheets["Ranking of rankings (τ_b)"]
        sheets_list = [sheet_Rankings, sheet_M, sheet_T, sheet_t_b, sheet_M_t_b, sheet_final_res]
        default_col_width = 8
        for sheet in sheets_list:
            # Установка базового стиля
            sheet.set_column("A1:XFD1048576", default_col_width, custom_format(writer, mode="base"))
        num_rows = rankings_all.shape[0]
        num_cols = rankings_all.shape[1]
        sheet_Rankings.set_row(0, 110, custom_format(writer, mode="header_col", rotation=90))
        sheet_Rankings.set_column("A1:A1", 50, custom_format(writer, mode="header_row"))
        sheet_t_b.set_column("A1:A1", 19, custom_format(writer, mode="header_row"))
        sheet_t_b.set_row(0, 110, custom_format(writer, mode="header_col", rotation=90))
        sheet_M_t_b.set_column("A1:A1", 19, custom_format(writer, mode="header_row"))
        sheet_M_t_b.set_row(0, 110, custom_format(writer, mode="header_col", rotation=90))
        sheet_final_res.set_column("B1:B1", 19, custom_format(writer, mode="base"))
    # Выгрузка результатов
    with open("Results.xlsx", "rb") as file:
        ste.download_button(
            label="Скачать результаты",
            data=file,
            file_name="Results.xlsx"
        )
    os.remove("Results.xlsx")
