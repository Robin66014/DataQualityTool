import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.markdown("# Page 2 ️")
st.sidebar.markdown("# Page 2")


if st.button("Next page"):
    switch_page("Page3")

if st.button("Previous page"):
    switch_page("app")