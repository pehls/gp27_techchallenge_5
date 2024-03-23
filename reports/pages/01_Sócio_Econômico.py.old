import streamlit as st
import pandas as pd
from src import get_data, train_model, generate_graphs

st.set_page_config(layout='wide')
st.title('Dados Sócio-Econômicos')

# add a box to upload  file
with st.sidebar.expander("Importar dados sócio-econômicos"):
    file = st.file_uploader('Selecione o arquivo (xlsx) com os ')
    load_file = st.checkbox("Carregar arquivo")

with st.sidebar.expander("Selecionar Tipo de Gráfico"):
    graph_type = st.selectbox("Tipos de Gráfico", generate_graphs.plot_style)

st.info("""
    Ao importar os dados, podemos conduzir diversas análises e plotar diferentes gráficos para nos ajudar a analisá-los. 
        
    Para simplificar, acrescente os nomes das colunas na primeira aba do excel, pois os usaremos para selecionar o dado a ser exibido!
    
    Os mesmos dados poderiam ser adicionados na análise do Modelo, afim de identificar se o mesmo tem relação com a evasão dos alunos, estatisticamente
    """, icon='ℹ️')
if load_file:
    if (file is None):
        st.warning("Não esqueça de selecionar o arquivo ou arrastar e soltar na barra lateral!", icon='❗')
    else:
        df = get_data._read_file(file)

        col1, col2, col3 = st.columns(3)

        with col1:
            x_col = st.selectbox("Eixo X", df.columns)

        with col2:
            y_col = st.selectbox("Eixo Y", df.columns)

        with col3:
            title = st.text_input("Título do Gráfico", "Um Título Sensacional!")

        st.plotly_chart(
            generate_graphs.plot_style.get(
                graph_type
                )(df, x_col, y_col, title)
            )
