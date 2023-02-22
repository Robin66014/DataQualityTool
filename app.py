#importing required libraries

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from stqdm import stqdm
from time import sleep
from deepchecks.tabular.checks import ColumnsInfo
from deepchecks.tabular.checks import ClassImbalance
import io
from streamlit_extras.switch_page_button import switch_page
from DataTypeInference import obtain_feature_type_table, createDatasetObject




def main():

    TEMPLATE_WRAPPER = """
        <div style="height:{height}px;overflow-y:auto;position:relative;">
            {body}
        </div>
        """

    # file upload button
    file = st.file_uploader("Please choose a .CSV file")  # TODO dit ook opslaan in cache als session state variable?

    option = st.selectbox('What type of data does your file consist of?', ('Structed tabular', 'Image/vision', 'Time-series'))
    st.write('You selected:', option)

    if st.button("Next page"):
        switch_page("Page2")

    if file is not None:
        #check file extensions TODO: fix for images and .ARFF
        if file.name.lower().endswith('.csv'):
            if "dataset" not in st.session_state:
                st.session_state.df = pd.read_csv(file)
        elif file.name.lower().endswith('.xlsx'):
            if "dataset" not in st.session_state:
                st.session_state.df = pd.read_excel(file)


        #TODO cache big variables in session_state object --> faster rerunning

        if "featureTypeTable" not in st.session_state:
            st.session_state.featureTypeTable = obtain_feature_type_table(st.session_state.df)

        st.table(st.session_state.featureTypeTable)
        dropDownOptions = ['NOT APPLICABLE'] #create list of dropdown options, initialize with no target column
        dropDownOptions.extend(list(st.session_state.df.columns)) #add column names to dropdown menu

        st.session_state.targetColumn = st.selectbox(label = 'Select your label/target column', options = dropDownOptions)

        if "dataset" not in st.session_state:
            st.session_state.dataset = createDatasetObject(st.session_state.df, st.session_state.featureTypeTable, st.session_state.targetColumn)


        # todo  LET OP: dit is waarschijnlijk het punt waar de package samenkomt met de web app

        #todo run checks based on dataset object created on this page upon pressing next page button / first thing that is done on page 2






        # check1 = ColumnsInfo()
        # st.write("Thesis Proposal Example using Deepchecks")
        # check1_result = check1.run(df)
        #
        # string_io = io.StringIO()
        # check1_result.save_as_html(string_io)
        # result_html = string_io.getvalue()
        #
        # for _ in stqdm(range(10), desc="Please wait patiently while your Data Readiness Report is being generated", mininterval=1):
        #     sleep(0.5)
        #
        # if result_html:
        #     height_px = 300
        #     html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        #     #with result_col:
        #     components.html(html, height=height_px)
        #
        # check2 = ClassImbalance()
        # check2_result = check2.run(df)
        # string_io = io.StringIO()
        # check2_result.save_as_html(string_io)
        # result2_html = string_io.getvalue()
        # if result2_html:
        #     height_px = 700
        #     html = TEMPLATE_WRAPPER.format(body=result2_html, height=height_px)
        #     #with result_col:
        #     components.html(html, height=height_px)
        # #st.write("Deepcheck check Type: " + name)
        # #jsonResult.render_deepchecks_test_result(results, dc_selection)
        # #st.write(bytes_data)


if __name__ == "__main__":
    # page layout
    page_title = 'Data quality webapp'
    page_icon = ':chart_with_upwards_trend:'
    layout = 'centered'
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout, initial_sidebar_state="collapsed")
    st.title(page_title + " " + page_icon)
    main()
