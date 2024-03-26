import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src import get_data

df_base = get_data._df_passos_magicos()

st.write("""
    # Tech Challenge #05 - Grupo 27 
    ## Passos Mágicos / 
    by. Eduardo Gomes, Igor Brito e Gabriel Pehls
""")
st.info(f"""
    Com objetivo de predizer a evasão e identificar o que influencia a mesma, mostramos nesse trabalho
    todo o processo para criação do nosso modelo, e algumas análises do histórico do mesmo.
    
    Os dados aqui utilizados foram recebidos da Passos Mágicos
    e contemplam o período de {min(df_base.dropna(axis=0, subset=['ULTIMO_ANO']).ULTIMO_ANO)} até {max(df_base.dropna(axis=0, subset=['ULTIMO_ANO']).ULTIMO_ANO)}.
""")
        

st.divider()
