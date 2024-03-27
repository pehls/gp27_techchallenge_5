import streamlit as st
from src import get_data, train_model, generate_graphs
import os, time
st.title('Simulação')

# df_base = get_data._df_passos_magicos()
df_model, cols = get_data._get_modelling_data(get_data._df_passos_magicos())
model_response_train = (train_model._run_xgboost(df_model, predict=True, retrain=False))
 

st.markdown(f"""             
    #### Utilização dos modelos
            """)
with st.expander("Instruções"):
    st.markdown(f""" 
        Com todos os dados em mãos, podemos criar um modelo para identificar possíveis alunos que irão acabar saindo da organização, antevendo e agindo de forma pró-ativa para retê-los, e organizando estrategicamente a estrutura da Passos Mágicos.

        Nesta página, iremos mostrar as variáveis que induzem ao resultado predito, através de alguns gráficos, com o apoio de uma biblioteca de códigos chamada Explainer Dashboard.
                
        Para tal, precisamos de um arquivo com a seguinte estrutura:
                """)
    df_example = get_data._read_file('data/processed/dado_simulacao.xlsx').head(5)

    st.table(df_example)

    get_data._return_download_data(df_example)

    file = st.file_uploader(
        'Selecione o arquivo (xlsx) com os dados dos alunos em Y-1 e Y (ano atual)', 
        type=['xlsx','application/vnd.ms-excel']
        )
col1, col2 = st.columns(2)
with col1:
    load_file = st.checkbox("Carregar arquivo")
with col2:
    if st.button("Resetar Simulações (Caso os detalhes não apareçam abaixo)"):
        st.components.v1.iframe('http://localhost:8050/quit/', width=0, height=0)

if file is not None:
    response_new_data = get_data._load_new_data(file)
    if (load_file):
        if (response_new_data['status_ok']):
            df_new_data = response_new_data['df']
           
        else:
            st.warning(f"Erro nas colunas {','.join(response_new_data['cols_diff'])}")
if (file is None) and load_file:
    st.warning("Não esqueça de carregar o arquivo nas Instruções!")
if (file is not None) and load_file:
    resp = generate_graphs._expose_explainer_custom_dashboard(
            model_response_train, 
            df_new_data=df_new_data
            )
    # st.write(resp)
    st.components.v1.iframe('https://0.0.0.0:8050/explainer_dashboard/', width=1200, height=900, scrolling=True)

# if (file is not None) and load_file:
#     while not(os.path.isfile("file.html")):
#         time.sleep(5)
#     with open("file.html") as html_file: 
#         html_data = html_file.read()
#     st.components.v1.iframe(
#         html_data,
#         width=None, height=900, scrolling=True
#       )