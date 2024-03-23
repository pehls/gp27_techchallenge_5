import streamlit as st
import pandas as pd

st.set_page_config(layout='wide')
st.title('Dados Sócio-Econômicos')

file = st.file_uploader('Selecione o arquivo (xlsx) com os dados sócio-econômicos')
if file is not None:
    dados = pd.read_excel(file, sheet_name=None)
    dados['relação das variáveis'].set_index('Código da variável', inplace=True)
    colunas = dados['relação das variáveis'].to_dict()['Descrição da variável']
    dados['PSE2020_domicílios'].drop(columns=['filter_$'], inplace=True)
    dados['PSE2020_domicílios'].rename(columns=colunas, inplace=True)
    dados = dados['PSE2020_domicílios']

    st.write(dados)

with st.sidebar:
    st.title('Configurações')
    st.selectbox('Selecione a coluna', dados.columns)