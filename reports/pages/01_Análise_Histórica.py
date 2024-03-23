import streamlit as st
import config
from src import get_data, train_model, generate_graphs

st.title('Análise Histórica')

df_base = get_data._df_passos_magicos()
df_boruta = get_data._df_boruta_shap(df_base)

tab_var_importance = st.tabs(['Importância das variáveis'])

with tab_var_importance:
    st.markdown(f"""
    """)