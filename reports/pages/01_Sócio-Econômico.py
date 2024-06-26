import streamlit as st
import pandas as pd
import time

from src import generate_graphs


st.set_page_config(layout='wide')
st.title('Dados Sócio-Econômicos')

st.info("""
    Ao importar os dados, podemos conduzir diversas análises e plotar diferentes gráficos para nos ajudar a analisá-los. 
        
    Para simplificar, acrescente os nomes das colunas na primeira aba do excel, pois os usaremos para selecionar o dado a ser exibido!
    """, icon='ℹ️')

def load_data():
    return pd.read_csv('data/processed/passos-socio-economico.csv')

if "file_key" not in st.session_state:
    st.session_state['file_key'] = 0

if "dados" not in st.session_state:
    st.session_state['dados'] = None

with st.sidebar:
    with st.container(border=True,):
        st.expander('Selecione o arquivo (xlsx) com os dados sócio-econômicos')
        file = st.file_uploader('', type=['xlsx'], label_visibility='collapsed', key=st.session_state['file_key'])
        if file is not None:
            try:
                dados = pd.read_excel(file, sheet_name=None)
                dados['relação das variáveis'].set_index('Código da variável', inplace=True)
                colunas = dados['relação das variáveis'].to_dict()['Descrição da variável']
                dados['PSE2020_domicílios'].drop(columns=['filter_$'], inplace=True)
                dados['PSE2020_domicílios'].rename(columns=colunas, inplace=True)
                dados = dados['PSE2020_domicílios']
                st.session_state['dados'] = dados
            except:
                msg = st.error(f'Arquivo **"{file.name}"** inválido', icon='⛔')
                time.sleep(15)
                msg.empty()
        st.write('OU carregue da memória')
        if st.button('Carregar dados da memória', type='primary', use_container_width=True,):
            st.session_state['dados'] = load_data()
        if st.button('Limpar dados', use_container_width=True,):
            st.session_state['dados'] = None
            st.session_state['file_key'] += 1
            st.rerun()

dados = st.session_state['dados']
if dados is not None:
    dados = dados.drop(columns=['Código do domicilio'])
    with st.container(border=True,):
        x = st.selectbox('Eixo X - Selecione a coluna', dados.columns.sort_values())
        st.write(f'Valores na coluna: {list(dados[x].unique())}')
                    
    if st.button("Plotar Gráfico"):
        st.plotly_chart(
            generate_graphs._histogram_plot(dados, x, title=f'Histograma - {x}'),
            use_container_width=True,
            )
