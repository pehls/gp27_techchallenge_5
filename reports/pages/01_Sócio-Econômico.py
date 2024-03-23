import streamlit as st
import pandas as pd
import time

st.set_page_config(layout='wide')
st.title('Dados Sócio-Econômicos')

st.info("""
    Ao importar os dados, podemos conduzir diversas análises e plotar diferentes gráficos para nos ajudar a analisá-los. 
        
    Para simplificar, acrescente os nomes das colunas na primeira aba do excel, pois os usaremos para selecionar o dado a ser exibido!
    
    Os mesmos dados poderiam ser adicionados na análise do Modelo, afim de identificar se o mesmo tem relação com a evasão dos alunos, estatisticamente
    """, icon='ℹ️')

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
    st.title('Configurações do Gráfico')
    if dados is not None:
        x = st.selectbox('Eixo X - Selecione a coluna', dados.columns)
        y = st.selectbox('Eixo Y - Selecione a coluna', dados.columns)
        title = st.text_input("Título do Gráfico", "Um Título Sensacional!")