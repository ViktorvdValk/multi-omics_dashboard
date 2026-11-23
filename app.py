"""Multi-omics dashboard for studies 2230, 1947 and 1806.   Run:  streamlit run app.py"""
import streamlit as st

from views import ceramides, data_setup, energy, explore, integrated, skin_gut

st.set_page_config(page_title="Skin multi-omics", layout="wide")

# Streamlit forgets the value of a widget when its page is not shown. The file locations of Data setup
# (keys cfg_*) are written back on every run, so they survive a visit to another page.
for _k in [k for k in st.session_state if str(k).startswith("cfg_")]:
    st.session_state[_k] = st.session_state[_k]

pages = {
    "Data": [st.Page(data_setup.render, title="Data setup", url_path="data", default=True)],
    "Single modality": [
        st.Page(skin_gut.skin_page, title="Skin microbiome", url_path="skin"),
        st.Page(skin_gut.gut_page, title="Gut microbiome", url_path="gut"),
        st.Page(energy.render, title="Energy metabolism", url_path="energy"),
        st.Page(ceramides.render, title="Lipidomics (ceramides)", url_path="ceramides"),
    ],
    "Across modalities": [st.Page(integrated.render, title="Integrated analysis", url_path="integrated"),
                          st.Page(explore.render, title="Subset explorer", url_path="explore")],
}
st.navigation(pages).run()
