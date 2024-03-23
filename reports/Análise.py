import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import src.generate_graphs as generate_graphs
import src.get_data as get_data

df_base = get_data._df_passos_magicos()

st.write("""
    # Tech Challenge #05 - Grupo 27 
    ## Passos Mágicos / 
    by. Eduardo Gomes, Igor Brito e Gabriel Pehls
""")
st.info(f"""
    Com objetivo de predizer o valor do Petróleo Brent, mostramos nesse trabalho
    todo o processo para criação do nosso modelo, e algumas análises do histórico do mesmo.
    
    Os dados aqui utilizados foram recebidos da Passos Mágicos
    e contemplam o período de {min(df_base.dropna(axis=0, subset=['ULTIMO_ANO']).ULTIMO_ANO)} até {max(df_base.dropna(axis=0, subset=['ULTIMO_ANO']).ULTIMO_ANO)}.
""")
        

tab_socio_economicos, tab_correlacoes = st.tabs(['Dados Socioeconômicos','Correlações', ])

with tab_socio_economicos:
    st.markdown(f"""
    """)

    
with tab_correlacoes:

    st.markdown(f"""
    """)
    st.plotly_chart(
        generate_graphs.plot_corr(df_base)
    )

    st.divider()

    st.markdown(f"""
    """)