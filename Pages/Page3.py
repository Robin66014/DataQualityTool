import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.markdown("# Page 3")
st.sidebar.markdown("# Page 345")


if st.button("Next page"):
    switch_page("Page3")

if st.button("Previous page"):
    switch_page("app")