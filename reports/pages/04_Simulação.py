import streamlit as st
st.set_page_config(layout="wide")

from src import get_data, train_model, generate_graphs
import os, time
import socket
from explainerdashboard import ExplainerDashboard
import numpy as np

st.title('Simulação')



# df_base = get_data._df_passos_magicos()
df_model, cols = get_data._get_modelling_data(get_data._df_passos_magicos())
df_example = get_data._load_new_data('data/processed/dado_simulacao.xlsx')['df']
model_response_train = (train_model._run_xgboost(df_model, predict=True, retrain=False))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
explainer = None

st.markdown(f"""             
    #### Utilização dos modelos
            """)
with st.expander("Instruções"):
    st.markdown(f""" 
        Com todos os dados em mãos, podemos criar um modelo para identificar possíveis alunos que irão acabar saindo da organização, antevendo e agindo de forma pró-ativa para retê-los, e organizando estrategicamente a estrutura da Passos Mágicos.

        Nesta página, iremos mostrar as variáveis que induzem ao resultado predito, através de alguns gráficos, com o apoio de uma biblioteca de códigos chamada Explainer Dashboard.
                
        Para tal, precisamos de um arquivo com a seguinte estrutura:
                """)

    st.table(df_example.head(5))

    get_data._return_download_data(df_example)

    file = st.file_uploader(
        'Selecione o arquivo (xlsx) com os dados dos alunos em Y-1 e Y (ano atual)', 
        type=['xlsx','application/vnd.ms-excel']
        )
col1, col2 = st.columns(2)
with col1:
    load_file = st.button("Carregar arquivo")
with col2:
    reload = st.button("Recarregar Simulações (Caso os detalhes não apareçam abaixo)")

if file is not None or reload:
    response_new_data = get_data._load_new_data(file)
    if (response_new_data['status_ok']):
        st.session_state['df_new_data'] = response_new_data['df']
        explainer = train_model.get_explainer(model_response_train, st.session_state['df_new_data'])
    else:
        st.warning(f"Erro nas colunas {','.join(response_new_data['cols_diff'])}")

    st.divider()
    
    # criar selectbox pro aluno
    _aluno = st.selectbox("Selecione um aluno", st.session_state['df_new_data'].index)

    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        with st.container(border=True):
            st.markdown("""## Contribuições
**Como cada Variável contribuiu para a previsão?**
                        
Este gráfico mostra a contribuição que cada característica individual tem na previsão de uma observação 
específica. As contribuições (a partir da média da população) somam-se à previsão final. Isso permite 
explicar exatamente como cada previsão individual foi construída a partir de todos os ingredientes 
individuais do modelo.""")

            st.plotly_chart(
                explainer.plot_contributions(_aluno)
                , use_container_width=True
            )
    with col2:
        with st.container(border=True):
            st.markdown("""## Previsões & Probabilidades
Mostra a probabilidade predita para cada situação, sendo False para não-evasão e True para Evasão.
""")
            st.plotly_chart(
                explainer.plot_prediction_result(_aluno)
                , use_container_width=True
            )

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.markdown("""## Dependência Shap
**Relação entre o valor da variável e o valor SHAP**
                        
Este gráfico mostra a relação entre as variáveis e seu valor shap, 
fazendo com que consigamos investigar o relacionamento entre o valor das features e 
o impacto na previsão. Você pode checar se o modelo usa as features de acordo com o
esperado, ou usar o mesmo para aprender as relações que o modelo aprendeu entre
as features de entrada e a saída predita. Em cinza, o NOME selecionado!                          
""")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                _coluna1 = st.selectbox(
                    "Selecione a Coluna (eixo X)", 
                    model_response_train['pipeline'][:-1].get_feature_names_out()
                    )

            with subcol2:
                _coluna2 = st.selectbox(
                    "Selecione a Coluna (cores)", 
                    list(set(model_response_train['pipeline'][:-1].get_feature_names_out()).difference(set([_coluna1])))
                    )
            
            st.plotly_chart(
                explainer.plot_dependence(col=_coluna1, color_col=_coluna2, highlight_index=_aluno)
                , use_container_width=True
            )
    with col4:
        with st.container(border=True):
            st.markdown("""## Gráfico de Dependência Parcial
**Como a previsão mudará se você mudar uma variável?**

O gráfico de dependência parcial (pdp) mostra como seria a previsão do modelo
se você alterar um recurso específico. O gráfico mostra uma amostra
de observações e como essas observações mudariam com as mudanças aplicadas
(linhas de grade). O efeito médio é mostrado em cinza. O efeito
de alterar o recurso para um único Nome é
mostrado em azul.                     
""")
            _coluna3 = st.selectbox(
                    "Selecione a Coluna:", 
                    model_response_train['pipeline'][:-1].get_feature_names_out()
                    )
            st.plotly_chart(
                explainer.plot_pdp(col=_coluna3, index=_aluno)
                , use_container_width=True
            )
                
else:
    st.warning("Não esqueça de selecionar o arquivo!")