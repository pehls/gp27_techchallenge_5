import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import src.generate_graphs as generate_graphs
import src.get_data as get_data

df_petroleo = get_data._df_petroleo()
st.write("""
    # Tech Challenge #05 - Grupo 27 
    ## Passos Mágicos / 
    by. Eduardo Gomes, Igor Brito e Gabriel Pehls
""")
         
st.info(f"""
    Com objetivo de predizer o valor do Petróleo Brent, mostramos nesse trabalho
    todo o processo para criação do nosso modelo, e algumas análises do histórico do mesmo.
    
    Os dados aqui utilizados foram baixados do site [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) 
    e contemplam o período de {min(df_petroleo.Date).date()} até {max(df_petroleo.Date).date()}.
""")
        

tab_grafico_historico, tab_seasonal, tab_adf, tab_acf = st.tabs([''])

with tab_adf:

    st.markdown(f"""
    Este é um teste estatístico utilizado na análise de séries temporais para determinar 
    se uma determinada série temporal é estacionária ou não.
    """)
    
    grafico_adf, series_adf = generate_graphs._adf(df_petroleo)
    res_adf = get_data._adfuller(series_adf)
    st.plotly_chart(
        grafico_adf,
        use_container_width=True,
    )

    st.markdown(f"""
        Aplicando a função `adfuller` ao dados sem nenhum tratamento, verificamos que a série não é estacionária
        ```
        Teste estatístico: {res_adf[0]}
        p-value: {res_adf[1]}
        Valores críticos: {res_adf[4]}
        ```
    """)

    st.divider()

    grafico_adf_diff, series_adf_diff = generate_graphs._adf_diff(df_petroleo)
    res_adf_diff = get_data._adfuller(series_adf_diff)
    st.plotly_chart(
        grafico_adf_diff,
        use_container_width=True,                                 
    )

    st.markdown(f"""
        Normalizando os dados com a diferenciação conseguimos transformar a série em estacionária.
        ```
        Teste estatístico: {res_adf_diff[0]}
        p-value: {res_adf_diff[1]}
        Valores críticos: {res_adf_diff[4]}
        ```
    """)