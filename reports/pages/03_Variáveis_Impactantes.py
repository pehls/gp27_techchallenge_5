import streamlit as st
from src import get_data, train_model, generate_graphs

st.title('Variáveis Impactantes')

tab_imp_variaveis, tab_shap_values, tab_simulacao = st.tabs(["Importância das Variáveis", "Como as Variáveis impactam a previsão?", "Simulação"])

df_base = get_data._df_passos_magicos()
df_model, cols = get_data._get_modelling_data(df_base)
df_boruta = get_data._df_boruta_shap(df_base)
model_response_train = (train_model._run_xgboost(df_model, predict=True, retrain=False))

with tab_imp_variaveis:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da necessidade de analisarmos de forma mais direta o que está influenciando na previsão da saída de um aluno, iremos utilizar alguns 
        itens adicionais para melhorar o desempenho de nossa previsão:
                
        ## Variáveis externas
                
        Com a descoberta de variáveis de diferentes fontes que tem uma relação forte com nossa variável alvo, vamos aproveitá-las para obter um modelo potencialmente melhor, mas principalmente mais explicativo;
        
        ### Feature Importance

        Ao utilizarmos diversas variáveis para prever a saída de um aluno, podemos mensurar qual a importância de cada uma delas para a previsão do mesmo.

        No caso apresentado a seguir, apresentamos a importância através do "gain", ou seja, o erro no treinamento reduzido em cada divisão, em todas as árvores, estando mais relacionado a como as árvores operam individualmente.

        Comumente, podemos definir as variáveis com maior importância como os mais impactantes para a movimentação da variável alvo, tornando a análise útil para identificar onde podemos atuar em uma melhoria possível para a variável alvo, norteando ações que possam causar o maior impacto possível em menos tempo;
                  
       

    """)
  
    st.plotly_chart(
        generate_graphs._plot_importance(df_model),
        height=400
    )

with tab_shap_values:
    st.markdown(f"""
    Para termos uma orientação de como as variáveis impactam, temos uma análise muito interessante:
    
    ### Shapley Values
    """)

    show_only=15
    height=show_only*40

    st.write(
        generate_graphs._plotly_plot_shap_values(
            response = model_response_train
            , width=800, height = height
            , show_only = show_only)
    )

    st.divider()
   
        
                
    st.markdown(f"""     
               
       
    """)
    
    st.markdown(f"""             
        #### Utilização dos modelos
                
        Com todos os dados em mãos, podemos criar um modelo para identificar possíveis alunos que irão acabar saindo da organização, antevendo e agindo de forma pró-ativa para retê-los, e organizando estrategicamente a estrutura da Passos Mágicos.
    """)