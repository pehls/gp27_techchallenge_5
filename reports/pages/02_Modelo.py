import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
from PIL import Image

st.title('Modelo')

tab_modelagem_inicial, tab_resultados_iniciais, tab_conceitos, tab_variaveis, tab_simulacao = st.tabs(['Modelagem', "Resultados", 'Conceitos', "Variáveis adicionais", "Simulação"])

df_base = get_data._df_passos_magicos()
df_model = get_data._get_modelling_data()
df_base = get_data._df_passos_magicos()
df_boruta = get_data._df_boruta_shap(df_base)

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
        ### Shapley Values
                
        ### Feature Importance

        Ao utilizarmos diversas variáveis para prever a saída de um aluno, podemos mensurar qual a importância de cada uma delas para a previsão do mesmo.

        No caso apresentado a seguir, apresentamos a importância através do "gain", ou seja, o erro no treinamento reduzido em cada divisão, em todas as árvores, estando mais relacionado a como as árvores operam individualmente.

        Comumente, podemos definir as variáveis com maior importância como os mais impactantes para a movimentação da variável alvo, tornando a análise útil para identificar onde podemos atuar em uma melhoria possível para a variável alvo, norteando ações que possam causar o maior impacto possível em menos tempo;
                
    """)
    
with tab_resultados_iniciais:
    
    st.plotly_chart(
        generate_graphs.plot_shap1(df_boruta)
    )
    st.plotly_chart(
        generate_graphs.plot_shap2(df_boruta)
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{baseline_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{baseline_r2}**
                
    """)

with tab_conceitos:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da necessidade de analisarmos de forma mais direta o que está influenciando na previsão da saída de um aluno, iremos utilizar alguns 
        itens adicionais para melhorar o desempenho de nossa previsão:
                
        #### Variáveis externas
                
        Com a descoberta de variáveis de diferentes fontes que tem uma relação forte com nossa variável alvo, vamos aproveitá-las para obter um modelo potencialmente melhor, mas principalmente mais explicativo;
                
       

    """)
    

    st.markdown(f"""     
               
       
    """)
    
    st.markdown(f"""             
        #### Utilização dos modelos
                
        Com todos os dados em mãos, podemos criar um modelo para identificar possíveis alunos que irão acabar saindo da organização, antevendo e agindo de forma pró-ativa para retê-los, e organizando estrategicamente a estrutura da Passos Mágicos.
    """)

with tab_variaveis:
    df_final = get_data._df_tree_modelling()
    dict_results = train_model._run_xgboost(df_final)
    df_importances = train_model._get_tree_importances(dict_results['pipeline'])
    st.markdown("""
    Para a análise de quais features mais importam, treinaremos um segundo modelo - chamado XGBoost, conforme explicado nos conceitos.
    Abaixo, vemos os passos do pipeline de previsão:
    """)
    st.image(config.PIPELINE,
                caption="Pipeline de Previsão usando XGBoost",
                width=680,
        )
    st.markdown(f"Para esse modelo, o mape ficou em **{dict_results['mape']}**!")

    st.plotly_chart(
        generate_graphs._plot_df_importances(df_importances),
        use_container_width=True
    )
    st.markdown("""
    Aqui, notamos a presença dominante das informações de exportação de combustíveis, de países diferentes, como Índia, Bolívia, Polônia, Sudão, Georgia, Alemanha, Israel, Dinamarca, Jordânia, Groenlândia, Canadá e Finlândia.
    <decorrer pq>.
                
    O valor médio da Dow jones também aponta como influente, sendo um dos indicadores apresentados anteriormente.
    """)

with tab_simulacao:
    st.markdown("Para efeito de simulação dos futuros preços do petróleo e do deploy de um modelo, vamos modificar duas das features organizadas pela importância do modelo de árvore (XGBoost), dado que possuam alguma importância no modelo:")
    col1, col2 = st.columns(2)
    with col1:
        option1 = st.selectbox(
            "Selecione a primeira:",
            (x.split('__')[1] for x in df_importances['Features'])
        )
    with col2:
        adjust1 = st.slider('Percentual (1)', -1.00, 1.00, 0.05)

    col1, col2 = st.columns(2)
    with col1:
        option2 = st.selectbox(
            "Selecione a Segunda:",
            (x.split('__')[1] for x in list(set(df_importances['Features'])-set([option1])))
        )
    with col2:
        adjust2 = st.slider('Percentual (2)', -1.00, 1.00, 0.05)

    st.divider()

    st.markdown('Para efeitos de simulação, vamos modificar conforme o solicitado, gerando as previsões a seguir:')
   
    # df_final = pd.DataFrame(imp.fit_transform(df_final), columns=df_final.columns).iloc[-10:]
    res = train_model.adjust_predict_data(
        df_final, 
        dict_cols = {
              option1:adjust1
            , option2:adjust2
        },
        _model=train_model._run_xgboost
    )
    st.plotly_chart(
        generate_graphs._plot_predictions(res['predictions']),
        use_container_width=True
    )
    st.markdown(f"A modificação das variáveis conforme selecionado, modificou em {res['mpe']} o valor do Petróleo, em média")
