import streamlit as st
import pandas as pd
import time

st.set_page_config(layout='wide')
st.title('Dados Sócio-Econômicos')

def load_data():
    return pd.read_csv('data/processed/passos-socio-economico.csv')

dados = None

if "file_key" not in st.session_state:
    st.session_state['file_key'] = 0

st.write('Selecione o arquivo (xlsx) com os dados sócio-econômicos')
file = st.file_uploader('', type=['xlsx'], label_visibility='collapsed', key=st.session_state['file_key'])
if file is not None:
    try:
        dados = pd.read_excel(file, sheet_name=None)
        dados['relação das variáveis'].set_index('Código da variável', inplace=True)
        colunas = dados['relação das variáveis'].to_dict()['Descrição da variável']
        dados['PSE2020_domicílios'].drop(columns=['filter_$'], inplace=True)
        dados['PSE2020_domicílios'].rename(columns=colunas, inplace=True)
        dados = dados['PSE2020_domicílios']
    except:
        dados = None
        msg = st.error(f'Arquivo **"{file.name}"** inválido', icon='⛔')
        time.sleep(15)
        msg.empty()
st.write('OU carregue clicando no botão abaixo')
if st.button('Carregar dados da memória', type='primary', use_container_width=True,):
    dados = load_data()
if st.button('Limpar dados', use_container_width=True,):
    dados = None
    st.session_state['file_key'] += 1
    st.rerun()
    
with st.sidebar:
    st.title('Configurações')
    if dados is not None:
        st.selectbox('Selecione a coluna', dados.columns)