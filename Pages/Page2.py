import streamlit as st
from streamlit_extras.switch_page_button import switch_page
#from app import dataset
from deepchecks.tabular.checks import ClassImbalance, ColumnsInfo
import io
import streamlit.components.v1 as components
from stqdm import stqdm
from time import sleep

st.markdown("# Page 2 ️")
st.sidebar.markdown("# Page 2")


st.write('Hello, welcome to page 2')


TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """


check2 = ColumnsInfo()
check2_result = check2.run(st.session_state.dataset)
string_io = io.StringIO()
check2_result.save_as_html(string_io)
result2_html = string_io.getvalue()
for _ in stqdm(range(10), desc="Please wait patiently while your Data Readiness Report is being generated", mininterval=1):
    sleep(0.5)
if result2_html:
    height_px = 700
    html = TEMPLATE_WRAPPER.format(body=result2_html, height=height_px)
    components.html(html, height=height_px)


with st.expander("Class imbalance"):
    st.write('The chart above shows some numbers I picked for you. I rolled actual dice for these, so theyre *guaranteed* to be random.')



if st.button("Next page"):
    switch_page("Page3")

if st.button("Previous page"):
    switch_page("app")