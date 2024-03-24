import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
from PIL import Image

import shap
shap.initjs()

st.title('Modelo')

tab_correlacoes, tab_modelagem_inicial, tab_resultados_iniciais = st.tabs(["Correlações",'Conceitos da Modelagem', "Resultados Iniciais"])

df_base = get_data._df_passos_magicos()
df_model, cols = get_data._get_modelling_data(df_base)
df_boruta = get_data._df_boruta_shap(df_base)

with tab_correlacoes:
    st.markdown("""
    Iniciaremos investigando quais variáveis tem um comportamento semelhante ao fato de termos uma evasão:
    """)
    st.plotly_chart(
        generate_graphs.plot_corr(df_base, show_only=['EVADIU'])
    )

with tab_modelagem_inicial:
    st.markdown("""
        Iniciando a etapa de modelagem, optamos por experimentar um modelo de árvore em conjunto com uma análise da importância e do sentido da influência das variáveis, através de algumas técnicas:
                
         ### XGBoost
        
        Um dos algoritmos de árvore conhecidos, se aproveita do resultado de diversas árvores de decisão, construídas de forma sequencial, onde cada nova árvore corrige o erro da árvore anterior, até que uma condição de parada seja alcançada.
        O algoritmo ainda aplica diversas penalidades de regularização, afim de evitar o overfitting, ou seja, uma adaptação muito forte aos dados de treinamento do modelo;
        """)
    st.image(config.DECISION_TREES,
                caption="Árvores de Decisão com Extreme Gradient Boosting (XGBoost)",
                width=700, 
        )   
    st.markdown(""" 
       
    """)
    
with tab_resultados_iniciais:
    model_response = train_model._run_xgboost(df_model, retrain=True, sampling=True)
    st.markdown("""
    Uma das formas de avaliarmos o resultado do modelo, é olhar para a própria matriz de confusão, que mostra se as nossas classificações foram corretas:
    
    ### Teste
                """)
    st.plotly_chart(
        generate_graphs._plot_confusion_matrix(
            model_response['confusion_matrix']
        )
    )
    model_response_train = (train_model._run_xgboost(df_model, predict=True, retrain=False))
    st.markdown(f"""
    ### Métricas
    Com este modelo, adquirimos uma precisão de **{model_response['precision']}**, um recall de **{model_response['recall']}** e uma área embaixo da curva roc de **{model_response['roc_auc_score']}**

    #### Mas, o que elas querem dizer?

    Quanto ao Precision, ele nos mostra o quanto dos preditos como "churn" realmente são churn; o Recall, mostra quanto dos atuais "churns" foram preditos da forma correta.

    Considerando que seria mais interessante termos um aluno considerado como churn, mesmo não sendo, e ele receber uma atenção especial, a métrica de Recall seria a mais interessante, representando que estaríamos atuando da forma correta em **{model_response['recall']}** deles!

    No geral, adquirimos uma precisão de **{model_response_train['precision']}**, um recall de **{model_response_train['recall']}** e uma área embaixo da curva roc de **{model_response_train['roc_auc_score']}**
    """)
