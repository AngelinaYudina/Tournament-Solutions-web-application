import sys
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Functions import init_ranking, M_T_generator, game_creator, sorting, CO_2, CO_3, UC_M, ES, E, MC_McK, MC_D

st.set_page_config(
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

EPS = sys.float_info.epsilon

# image = Image.open("icon.ico")   TO-DO: adds HSE icon

st.title("Библиометрическая информационная система оценки научных журналов")
st.write("Пожалуйста, загрузите свой файл. Если у вас нет готового файла, вы можете скачать пример данных и "
         "работать с ним.")
# Sample data downloader
with open("sample_data.csv", "rb") as file:
    st.download_button(
        label="Нажмите, чтобы скачать пример данных",
        data=file,
        file_name="sample_data.csv"
    )
# File uploader
if "file" not in st.session_state:
    st.session_state["file"] = None
file = st.file_uploader("Загрузите файл", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
st.session_state["file"] = file
# File reader
if file is not None:
    st.success("Файл успешно загружен!")
    try:
        df = pd.read_csv(file, delimiter=";")
    except:
        df = pd.read_excel(file)
    #st.write("Here is the sample of the uploaded data:")
    #st.dataframe(df.head())
    #st.write("Пожалуйста, подождите.")
    # Creating G from df
    df = df.set_index(df.columns[0])
    rankings_all = init_ranking(df)
    #st.dataframe(rankings_all.head())
    M, T, counter = M_T_generator(rankings_all)
    G = game_creator(M)
    st.write(f"Количество журналов = {df.shape[0]}. Количество ничьих = {counter}. Исходные ранжирования: "
             f"{list(df.columns)}")
    #st.dataframe(G)
    # Final ranking table
    tournament_solutions = [CO_2, CO_3, UC_M, ES, E, MC_McK, MC_D]
    sol_names = ["Copeland rule (2 v.)", "Copeland rule (3 v.)", "sorting by UC", "sorting by ES", "sorting by E",
                 "sorting by MC_McK", "sorting by MC_D"]
    for i in range(len(tournament_solutions)):
        ranking = sorting(G, tournament_solutions[i])
        rankings_all[sol_names[i]] = ranking
    st.write("Ранжирования журналов")
    st.dataframe(rankings_all)

    corr = rankings_all.corr(method='kendall')
    ind = corr.columns
    st.write("Значения коэффициента t_b")
    st.dataframe(corr)
    fig = plt.figure(figsize=(14.7, 7.27))
    #rcParams['figure.figsize'] = 14.7, 8.27
    sns.heatmap(corr,
                xticklabels=ind,
                yticklabels=ind,
                cmap='coolwarm',
                annot=True,
                fmt='.3g',
                linecolor='black',
                linewidth=1
                )
    #fig.savefig("t_b.png")
    #image = Image.open('t_b.png')
    #st.image(image)
    st.pyplot(fig)

    #corr = df.corr(method='kendall')
    #ind = corr.columns
    matrix = [[0 for _ in range(len(ind))] for _ in range(len(ind))]
    for i in range(len(ind)):
        for j in range(len(ind)):
            val_i = 0
            val_j = 0
            if i < j:
                # for k in range(len(ind)):
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
    #matrix_pd = pd.DataFrame(matrix, index=ind, columns=ind)

    res = pd.DataFrame(matrix, columns=ind, index=ind)
    sum_col = []
    for row in matrix:
        sum_row = sum(row)
        sum_col.append(sum_row)
    res["CO_2"] = sum_col
    st.write("Матрица строгого мажоритарного отношения (τ_b)")
    st.dataframe(res)
    #res_tb = deepcopy(res)

    pos = [x + 1 for x in range(len(ind))]
    res.sort_values(by="CO_2", ascending=False, inplace=True)
    ind_ranked = res.index
    final_ranking = pd.DataFrame(ind_ranked, columns=["Метод"], index=pos)
    rank = []
    counter = 1
    value_prev = res['CO_2'].values.max()
    for value in res['CO_2']:
        if value < value_prev:
            counter += 1
        rank.append(counter)
        value_prev = value
    final_ranking["Ранг"] = rank
    st.write("Ранжирование ранжирований")
    #final_ranking.rename(columns={"0": "Метод"}, inplace=True)
    final_ranking = final_ranking.set_index("Ранг")
    st.dataframe(final_ranking)
    #final_tb = final_ranking
