import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)

st.title('Análise Histórica')

df_base = get_data._df_passos_magicos()
df_boruta = get_data._df_boruta_shap(df_base)

tab_var_importance = st.tabs(['Importância das variáveis'])

with tab_var_importance:
    st.markdown(f"""
    """)
    st.plotly_chart(
        generate_graphs.plot_shap1(df_boruta)
    )
    st.plotly_chart(
        generate_graphs.plot_shap2(df_boruta)
    )
