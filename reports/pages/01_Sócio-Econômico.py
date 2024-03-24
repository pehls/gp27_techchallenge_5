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

    with st.container(border=True,):
        graph_type = st.selectbox("Tipo de Gráfico", generate_graphs.plot_style)

dados = st.session_state['dados']
if st.session_state['dados'] is not None:
    with st.container(border=True,):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.selectbox('Eixo X - Selecione a coluna', dados.columns)
        with col2:
            y = st.selectbox('Eixo Y - Selecione a coluna', dados.columns)
        with col3:
            title = st.text_input("Título do gráfico", "Título do gráfico")
        with col4:
            col31, col32  = st.columns(2)
            with col31:
                operation = None
                if st.checkbox("Agrupar dados?"):
                    with col32:
                        operation = st.selectbox("Operação", ["Soma","Média",'Mediana'])
                    
    if st.button("Plotar Gráfico"):
        st.plotly_chart(
            generate_graphs.plot_style.get(
                graph_type
                )(dados, x, y, operation, title)
            )
