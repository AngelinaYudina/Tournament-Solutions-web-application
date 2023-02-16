from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Multipage app",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# image = Image.open("icon.ico")   TO-DO: add HSE icon

st.title("Homepage")
st.write("Hi! Using this app you are able to sort journals with respect to bibliometric indicators using the theory of "
         "social choice. In order to start, please, upload your file.")
st.write("If you don't have file to work with, you can download the sample data:")
st.download_button("Download the sample data", sample_data)   # TO-DO: add sample data
# File uploader
if "file" not in st.session_state:
    st.session_state["file"] = None
file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
st.session_state["file"] = file
# File reader
if file is not None:
    st.success("File uploaded successfully!")
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_excel(file)
    st.write("Here is the sample of the uploaded data:")
    st.dataframe(df.head())
    st.write("Now you can move to the Results page. You can always come back to this page to work with another file.")
