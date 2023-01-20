#importing required libraries

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from stqdm import stqdm
from time import sleep
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.checks import ColumnsInfo
from deepchecks.tabular.checks import ClassImbalance
from deepchecks.tabular import Dataset
import io
from streamlit_extras.switch_page_button import switch_page
from DataTypeInference import obtain_feature_type_table
#import Deepchecksreport
#import jsonResult
from scipy.io import arff
#adding a file uploader
page_title = 'Data quality webapp'
page_icon = ':chart_with_upwards_trend:'
layout = 'centered'

st.set_page_config(page_title = page_title, page_icon = page_icon, layout=layout, initial_sidebar_state="collapsed")
st.title(page_title + " " + page_icon)


file = st.file_uploader("Please choose a .CSV file")
TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """


option = st.selectbox('What type of data does your file consist of?', ('Structed tabular', 'Image/vision', 'Time-series'))
st.write('You selected:', option)

# st.write('Deselect irrelevant checks for your data analysis problem')
# agree = st.checkbox('Missing values')
# agree = st.checkbox('Feature feature correlation')
# agree = st.checkbox('Class imbalance')
# agree = st.checkbox('Wrong labels')
import pickle as pkle
import os.path

if st.button("Next page"):
    switch_page("Page2")


if file is not None:

    #To read file as bytes
    #bytes_data = file.getvalue()
    #df = pd.read_csv('datasets\iris.csv')
    df = pd.read_csv(file)
    #df = Dataset(df, label='Species')
    featureTypeTable = obtain_feature_type_table(df)
    st.table(featureTypeTable)
    #data = arff.loadarff('datasets\dataset_194_eucalyptus.arff')
    #data = arff.loadarff(bytes_data)
    #df = pd.DataFrame(data[0]) #nodig voor .arff to pandas df
    #suite = data_integrity()
    check1 = ColumnsInfo()
    st.write("Thesis Proposal Example using Deepchecks")
    check1_result = check1.run(df)

    #name, results, result_keys = Deepchecksreport.render_dc_report(df)
    string_io = io.StringIO()
    check1_result.save_as_html(string_io)
    result_html = string_io.getvalue()

    for _ in stqdm(range(10), desc="This is a slow task", mininterval=1):
        sleep(0.5)

    if result_html:
        height_px = 300
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        #with result_col:
        components.html(html, height=height_px)
    
    check2 = ClassImbalance()
    check2_result = check2.run(df)
    string_io = io.StringIO()
    check2_result.save_as_html(string_io)
    result2_html = string_io.getvalue()
    if result2_html:
        height_px = 700
        html = TEMPLATE_WRAPPER.format(body=result2_html, height=height_px)
        #with result_col:
        components.html(html, height=height_px)
    #st.write("Deepcheck check Type: " + name)
    #jsonResult.render_deepchecks_test_result(results, dc_selection)
    #st.write(bytes_data)
