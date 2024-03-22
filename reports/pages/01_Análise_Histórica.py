import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)

st.title('Análise Histórica')

df_petroleo = get_data._df_petroleo()
df_conflitos_porpais = get_data._events_per_country()
df_conflitos_mundiais = get_data._events_globally()
tab_volatilidade, tab_conflitos_armados, tab_energia_consumo, tab_exportacao, tab_indices_eua = st.tabs(['Volatilidade', "Conflitos", 'Combustíveis Fósseis', 'Exportação de Comb.', 'Índices EUA'])

# https://www.linkedin.com/pulse/petr%C3%B3leo-uma-an%C3%A1lise-sobre-volatilidade-yu64c/?originalSubdomain=pt
# https://pt.wikipedia.org/wiki/Crises_do_petr%C3%B3leo
# https://acleddata.com/data-export-tool/
# https://data.worldbank.org/topic/energy-and-mining - uso de energia, consumo de combustiveis fosseis, exportacao
with tab_volatilidade:
    st.markdown("""
        Nesta seção, vamos tratar de analisar eventos históricos, e observar através dos dados e de alguns testes estatísticos, se os mesmos realmente possuem alguma relação com o aumento ou a diminuição do preço do petróleo;
        Para tal, alguns conceitos devem ser apresentados:
        
                    
        - **Correlação de Pearson**: também chamada de correlação produto-momento, mede o grau de correlação entre duas variáveis de escala métrica, no instante x, ou seja, de forma direta no mesmo ponto do tempo. Quão mais próxima de 1, seja negativo ou positivo, mais forte é, de acordo com o seu sentido. Caso muito próxima de 0, nota-se uma correlação fraca.
                    
        E, ainda, um teste de causalidade:
                    
        - **Teste de Causalidade de Granger**: É um teste estatístico que visa superar as limitações do uso de simples correlações entre variáveis, analisando o sentido causal entre elas, demonstrando que uma variável X "Granger Causa" Y caso os valores do passado de X ajudam a prever o valor presente de Y.
        Tipicamente, aplica-se um teste F sobre os erros da predição de Y pelo seu passado, e da predição de Y pelo seu passado e pela variável X, visando testar a hipótese de que Y é predito apenas por seus valores passados (H0) e se o passado de X ajuda a prever Y (H1); Um ponto importante, é a necessidade de dados históricos para treinamento dos modelos que trazem a análise de causalidade de granger!

                
    """)

with tab_conflitos_armados:
    subtab_conflitos_paises, subtab_tipo_evento_petroleo, subtab_correlacao = st.tabs(['Países em Conflitos', 'Tipos de evento e Petróleo', 'Correlação'])

    with subtab_conflitos_paises:
        st.markdown("""
        Os dados foram obtidos no portal [Armed Conflict Location & Event Data Project](https://acleddata.com/data-export-tool/), um portal que coleta, analista e mapeia crises globais, 
        salvando informações diversas sobre tais conflitos em diferentes locais.
                    
        Embora traga análises interessantes, não iremos conseguir gerar a análise de causalidade de granger, devido a pouca disponibilidade de histórico dessas bases.
                    
        Para começar, analisemos os países com mais fatalidades nos anos analisados:
        """)

        st.plotly_chart(
            generate_graphs._plot_conflitos_paises(df_conflitos_porpais),
            use_container_width=True,
        )
        st.markdown("""
        Será que são os mesmos países com maior exportação de petróleo no período? Será que existe alguma correlação entre as fatalidades e o preço do Petróleo?
                    
        Para respoder tais perguntas, precisamos entender os tipos de eventos que a plataforma entrega:
        - **Batalhas**: Basicamente, são confrontos armados, com atores estatais ou não, sejam governos ou grupos armados apenas;
        - **Violência contra civís**: são ataques, desaparecimentos forçados, sequestros e violências sexuais;
        - **Explosões / Violência Remoda**: aqui estão incluídos ataques aéreos, com mísseis e artilharias, explosões remotas, ataques com drones ou por via aérea (aviões, por exemplo), granadas e bombas suicidas.
        
        Notamos, por essas descrições, que são eventos precisamente violentos, muitos derivados de confrontos geopolíticos, e até mesmo derivados de protestos com maior incidência de violência, como o que vemos atualmente na Ucrânia. 
                    Por tal evento ter um cunho territorial específico entre a Ucrânia e Rússia, com o anexo da Criméia, uma região que produz petróleo e gás, por exemplo, ela entra na categoria de Batalhas, e poderia ter uma forte relação com o aumento do petróleo na região e no mundo, a depender da força da produção do mesmo; Sabemos que existem outros motivos que poderiam determinar tal confronto, mas vamos focar nos impactos no preço do petróleo e sua produção mundial;
        """)

    with subtab_tipo_evento_petroleo:
        df_conflitos_preco_normalizados = get_data._events_normalized_globally()
        st.plotly_chart(
            generate_graphs._plot_conflitos_tipo(df_conflitos_porpais),
            use_container_width=True,
        )
        st.markdown("""
        De forma geral, notamos que a quantidade de fatalidades costuma ser maior com explosões e eventos relacionados, seguido por batalhas. Em poucos meses, e em menor quantidade de fatalidades, temos os casos de violência contra civís. Vamos analisar a quantidade de eventos e fatalidades vs Preço do Petróleo Brent:
        """)
        st.plotly_chart(
            generate_graphs._plot_conflitos_tipo_e_petroleo(df_conflitos_preco_normalizados), use_container_width=True
        )
        st.markdown("""
        Visualmente, ao compararmos apenas o Preço vs quantidade total de eventos e de fatalidades, vemos poucos pontos relacionados, mas uma coerência bem forte entre os números no período de 2021 (começo do ano) e de 2022 (começo do ano), fato que não se repete em 2023 (notamos aqui, um aumento no número de eventos e fatalidades, mas o preço do petróleo não tem picos e vales tão bem estabelecidos). No decorrer do ano de 23, os meses de Março, Abril e Maio tem caracteristicas muito semelhantes nos três indicadores, com o preço em uma crescente no restante do ano, até o fim dos dados.
                    
        Outro ponto interessante é o número de explosões aparentar estar altamente relacionado com o Preço, mas será que o mesmo se repete quando analisado de forma estatística? Ou o comportamento visual apenas? Vamos analisar:
        """)
    
    with subtab_correlacao:
        st.markdown("""
       
                    
        """)
        
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(
                get_data._events_correlations(df_conflitos_preco_normalizados)
                ), use_container_width=True
        )
        st.markdown("""
        Analisando a correlação das variáveis com o Preço do Petróleo, vemos que todas elas estão com uma correlação abaixo de 0.4, isso indica uma correlação fraca, ou quase inexistente.
        """)
        # ref https://www.questionpro.com/blog/pt-br/correlacao-de-pearson/
        # ref https://community.revelo.com.br/primeiros-passos-no-dtw/
        # ref https://www.linkedin.com/pulse/causalidade-de-granger-guilherme-garcia/?originalSubdomain=pt

with tab_energia_consumo:
    subtab_uso_energia, subtab_consumo_comb_fosseis, subtab_crises_consumo, subtab_correlacao, subtab_causalidade = st.tabs(['Uso de Energia', 'Consumo de Comb. Fósseis', 'Crises e Consumo', 'Correlação', 'Causalidade'])
    
    with subtab_uso_energia:
        df_uso_energia_top10 = get_data._df_energy_use_top10()
        df_uso_energia, lista_2015_nulo = get_data._df_energy_use(list(df_uso_energia_top10['Country Name'].values))
        st.markdown(f"""
        Além da incidência de conflitos em relação ao Preço do Petróleo, vamos analisar a relação do uso de energia primária (combustível fóssil), antes da transformação para outros combustíveis de utilização final.
        O indicador aqui apresentado refere-se a produção interna mais importações e variações de estoque, menos as exportações e combustíveis fornecidos a navios e aeronaves utilizados no transporte internacional.
        O dado pode ser obtido em [Energy use (kg of oil equivalent per capita)](https://data.worldbank.org/indicator/EG.USE.PCAP.KG.OE);
                    
        Primeiramente, vamos analisar quais são os países com maior uso, nos 5 anos mais recentes de dados (entre 2011 e 2015):
        """)
        st.plotly_chart(
            generate_graphs._plot_top10_recent_energy_use(df_uso_energia_top10),
            use_container_width=True
        )
        st.markdown(f"""
        Por intuição, pontuaríamos os Estados Unidos como um dos 5 maiores, mas podemos perceber que ele pontua apenas em 10º lugar. Interessante ainda, ver Trinidade e Tobago e Curacao como membros do top5.
        """)
    
    with subtab_crises_consumo:
        st.markdown("""
        Continuando, vamos verificar como esses Top 10 se relacionam com o Preço do Petróleo, observando um período anterior a 2015. Para tal, preenchemos o valor do indicador no ano de 2015, onde o mesmo era nulo, para os países:               
        """)
        
        for x in lista_2015_nulo:
            st.markdown(x)

        st.markdown(f"""
        Ainda, para facilitar a visualização de variáveis em diferentes unidades, os valores foram normalizados pela máxima e mínima, pontuando como 0 quando no mínimo, e 1 no máximo, considerando em separado o preço do Petróleo e os dados de uso de energia nos países;
        """)
        st.plotly_chart(
            generate_graphs._plot_energy_use(df_uso_energia),
            use_container_width=True
        )
        st.markdown(f"""
        Aqui, podemos destacar a proximidade dos preços do petróleo, e os valores do indicador nos países de Trinidad and Tobago e Brunei Darussalam entre os anos 1987-1997;
        entre os anos 1998/1999, o valor do preço do Petróleo aparenta ser inversamente proporcional ao uso de Curacao, por exemplo, que se amplifica fortemente;
        A partir de 1999, até 2008, vemos uma crescente bem forte do preço, praticamente em linha reta; da mesma forma, o indicador da Islândia (Iceland) se torna extremamente alto, se aproximando do valor máximo, que é admitido pelo Qatar em 2004; 
        A utilização de combustíveis fósseis tem um vale bem forte, juntamente com o seu preço, [no ano de 2009](https://www.politize.com.br/crise-financeira-de-2008/), período de uma crise mundial em decorrência da falência do banco de investimento norte-americano Lehman Brothers, que provocou uma [recessão econômica global](https://repositorio.ipea.gov.br/bitstream/11058/6248/1/RTM_v3_n1_Cinco.pdf), levando uma cadeia de falências de outras grandes financeiras.
        Em uma rápida recuperação, temos um novo pico, e o recorde para o período para o preço do petróleo em 2011, seguido por um aumento considerável da utilização do mesmo para países como Qatar, Curacao, Brunei Darussalam e os demais, seguidos por uma regularização do preço do petróleo em 2015, mas sem impactos significativos na utilização do mesmo para os países. 
        Neste ano de 2015, tendo iniciado em 2014 e findado em 2017, acontece uma nova crise econômica, com alguns fatores como o [fim do "superciclo das commodities"](https://brasil.elpais.com/brasil/2015/10/18/internacional/1445174932_674334.html), bem como uma desaceleração da economia chinesa, que vinha auxiliando a recuperação global desde 2008, e ainda uma ressaca econômica derivada do endividamento de muitos países europeus, [e até mesmo no Brasil](https://agenciabrasil.ebc.com.br/economia/noticia/2017-09/editada-embargo-10h-queda-de-2015-interrompeu-ascensao-do-setor-de-servicos).   
        """)
        st.image(config.PETR4_HISTORIC,
                caption="Crises do Petróleo",
                width=600,
        )
        st.markdown("""
        Conforme algumas fontes, [como esta](https://pt.wikipedia.org/wiki/Crises_do_petr%C3%B3leo), muitas dessas crises do petróleo foram motivadas por processos que geraram déficit de oferta, sejam conflitos, nacionalizações, crises políticas em países exportadores ou com economia altamente dependente do petróleo.
        """)

    with subtab_consumo_comb_fosseis:
        df_fuel_cons = get_data._df_fossil_fuel_cons()
        
        st.markdown(f"""
        Após dissertarmos sobre o uso de Energia Primária, vamos analisar o Consumo de Combustíveis Fósseis dos países, buscando identificar uma relação entre o uso per capita e o quanto os combustíveis fósseis representam, em percentual, da utilização derivada da matriz energética de cada país; 
        O dado pode ser obtido em [Fossil fuel energy consumption](https://data.worldbank.org/indicator/EG.USE.COMM.FO.ZS);
        """)
        st.write(df_fuel_cons)

        st.markdown("""
        Notamos aqui, que a maioria dos países está na região do meio-oeste e norte da África, países produtores de Petróleo; as exceções, Gibraltar e Brunei Darussalam, produzem e exportam Petróleo, em regiões diferentes.
        """)

    with subtab_correlacao:
        df_fuel_corr_causa = get_data._get_fossil_fuel_cons_energy_use_corr()
        st.markdown("""
        Para analisarmos a correlação entre os dois indicadores e o preço do petróleo, para cada país, no decorrer dos anos, vamos filtrar linhas (Anos) e colunas (indicadores de consumo e uso de energia/combustíveis fósseis), de forma a termos dados numéricos em ao menos 50% das mesmas;
        Visando simplificar a visualização, vamos selecionar uma faixa dela:
        """)
        faixa_corr = st.slider('Threshold ', 0.0, 1.0, 0.93)
        df_correlacoes = get_data._events_correlations(df_fuel_corr_causa, cols_to_plot=list(set(df_fuel_corr_causa.columns) - set(['Year'])))
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(
                df_correlacoes.loc[(abs(df_correlacoes[['Preco']]) >= faixa_corr)['Preco']][df_correlacoes.loc[(abs(df_correlacoes[['Preco']]) >= faixa_corr)['Preco']].index]
                ), use_container_width=True
        )
        faixas_correlacao = get_data._get_faixas_correlation(df_correlacoes)
        st.markdown(f"""
        Como temos muitos paises, resumidamente, temos, comparando a série de Preços:
        """)
        for x in faixas_correlacao:
            st.markdown('- '+str(faixas_correlacao[x]) + x) 

    with subtab_causalidade:
        st.markdown("""
        Para a análise de Causalidade, vamos preencher os nulos com a média, e testar com a Causalidade de Granger, de modo a filtrar apenas as variáveis que tiverem uma significância abaixo do limite escolhido abaixo, com o padrão em 5%:
        """)

        faixa_causalidade = st.slider('Threshold ', 0.01, 1.00, 0.05)
        
        df_causality = train_model.check_causality(
              df_fuel_corr_causa.dropna(axis=0, thresh=0.7).dropna(axis=1)
            , list_of_best_features=list(set(df_fuel_corr_causa.dropna(axis=0, thresh=0.7).dropna(axis=1).columns)-set('Preco'))
            , y_col='Preco'
            , threshold=faixa_causalidade
            )
        
        st.markdown('Para o nível de significância escolhido, as seguintes variáveis tem um efeito de causalidade de granger no Preço do Petróleo:')
        
        for _tuple in df_causality.itertuples():
            st.markdown(f'- {_tuple.Variable}')

with tab_exportacao:

    subtab_exportacao, subtab_correlacao, subtab_causalidade = st.tabs(['Exportação de Combustíveis','Correlação', 'Causalidade'])

    with subtab_exportacao:
        df_fuel_exp = get_data._df_fuel_exports()
        st.markdown("""
        Assim como o Consumo de Combustíveis fósseis, o dado da porcentagem da exportação de combustíveis, incluindo combustíveis minerais, lubrificantes e materiais relacionados, 
        está disponívei no world bank, através do [Fuel Exports (% of merchandise exports)](https://data.worldbank.org/indicator/TX.VAL.FUEL.ZS.UN?end=2022&start=2022&type=shaded&view=map&year=2022).
                    
        As estimativas são feitas através da plataforma WITS da base de dados Comtrade mantida pela Divisão de Estatística das Nações Unidas.
        """)
        st.write(df_fuel_exp)

        st.markdown("""
        Assim como na utilização, temos 4 países do meio-oeste/norte da África (Libia, Kuwait, Qatar, Emirados Árabes Unidos), e mais 3 países da África SubSaariana (Angola, Nigeria, Rep. do Congo). Fechando o top 10, temos ainda Brunei Darussalam, Azerbaijão e Timor-Leste. 
        Tais países possuem mais de 67% de sua economia vinculada a combustíveis fósseis. sendo que o top 5 está muito próximo ou acima de 90% de sua economia vinculada ao Petróleo.
        """)
    
    with subtab_correlacao:
        df_fuel_exp_corr = get_data._df_fuel_exp_corr()
        st.markdown("""
       
                    
        """)
        faixa_corr_fuel_exp = st.slider(' Threshold ', 0.0, 1.0, 0.93)
        df_corr_fuel_exp = get_data._events_correlations(df_fuel_exp_corr)
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(
                df_corr_fuel_exp.loc[(abs(df_corr_fuel_exp[['Preco']]) >= faixa_corr)['Preco']][df_corr_fuel_exp.loc[(abs(df_corr_fuel_exp[['Preco']]) >= faixa_corr)['Preco']].index]
                ), use_container_width=True
        )
        faixas_correlacao_fuel_exp = get_data._get_faixas_correlation(df_corr_fuel_exp)
        st.markdown(f"""
        Destacamos aqui, alguns países com uma correlação perfeita negativa (como Guiné-Bissau, Ilhas Cayman e Antigua e Barbuda), países que tem uma dependência econômica muito forte com o Petróleo, e positiva extremamente alta, como Canada, República da Corea, India, Afeganistão, Itália, Suécia e Tuvalu.
                    
        Como temos muitos paises, resumidamente, temos, comparando a série de Preços:
        """)
        for x in faixas_correlacao_fuel_exp:
            st.markdown('- '+str(faixas_correlacao_fuel_exp[x]) + x) 
            
    with subtab_causalidade:
        st.markdown("""
        Para a análise de Causalidade, vamos preencher os nulos com a média, e testar com a Causalidade de Granger, de modo a filtrar apenas as variáveis que tiverem uma significância abaixo do limite escolhido abaixo, com o padrão em 5%:
        """)

        faixa_causalidade_exp = st.slider(' Threshold  ', 0.01, 1.00, 0.05)
        colunas_causa = [x for x in list(set(df_fuel_exp_corr.dropna(axis=0, thresh=0.7).dropna(axis=1).columns)-set(['Preco','Year'])) if x.startswith('minmax')]
        df_causality = train_model.check_causality(
              df_fuel_exp_corr.dropna(axis=0, thresh=0.7).dropna(axis=1)
            , list_of_best_features=colunas_causa
            , y_col='Preco'
            , threshold=faixa_causalidade_exp
            )
        
        st.markdown('Para o nível de significância escolhido, as seguintes variáveis tem um efeito de causalidade de granger no Preço do Petróleo:')
        
        for _tuple in df_causality.itertuples():
            st.markdown(f'- {_tuple.Variable}')
            

with tab_indices_eua:

    st.markdown("""
    A ideia dessa análise é comparar e verificar correlação entre o preço do petróleo com dois dos principais índices financeiros dos EUA.

    Os dados foram recuperados no IpeaData e podem ser acessados através de [Dow Jones](http://www.ipeadata.gov.br/ExibeSerie.aspx?serid=40279&module=M) e
    [Nasdaq](http://www.ipeadata.gov.br/ExibeSerie.aspx?serid=603335808&module=M).  

    *Para melhor visualização gráfica, os dados foram normalizados.* 
    """)

    subtab_dowjones, subtab_nasdaq = st.tabs(['Dow Jones', 'Nasdaq'])
    df_dowjones, df_nasdaq = get_data._df_brent_dowjones_nasdaq()

    with subtab_dowjones:
        st.plotly_chart(
            generate_graphs._plot_index(df_dowjones, label_index='Dow Jones', label_period='Histórico'),
            use_container_width=True
        )

        st.markdown("""
        Observando a evolução histórica do preço de petróleo tipo Brent e comparando com a evolução do índice Dow Jones, não fica clara uma 
        correlação existente.
                    
        Contudo, nota-se que a crise econômica de 2008 e a pandemia da Covid-19 afetaram negativamente ambos os valores.
        """)

        corr_dowjones = df_dowjones.corr()
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(corr_dowjones).update_layout(title=f'Correlação Brent x Dow Jones'),
            use_container_width=True,
        )

        st.markdown(f"""
        Calculando a correlação através do método *Pearson* obtemos o valor de $p$ em **{corr_dowjones.loc['Brent', 'Dow Jones']:.4f}** e 
        pode ser interpretada como uma correlação positiva moderada.
        
        Apenas para avaliar o comportamento observado no gráfico histórico (crise e pandemia) e considerando os dados mais 
        recentes (a partir de 2005-01-01), temos:
        """)

        st.plotly_chart(
            generate_graphs._plot_index(df_dowjones.loc[df_dowjones.index >= '2005-01-01'], label_index='Dow Jones', label_period='A partir de 2005-01-01'),
            use_container_width=True
        )

        corr_dowjones_partial = df_dowjones.loc[df_dowjones.index >= '2005-01-01'].corr()
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(corr_dowjones_partial).update_layout(title=f'Correlação Brent x Dow Jones'),
            use_container_width=True,
        )

        st.markdown(f"""
        Com um intervalo menor nos dados obtivemos um valor $p$ **{corr_dowjones_partial.loc['Brent', 'Dow Jones']:.4f}** e demonstra 
        uma correlação (positiva/negativa) praticamente inexistente.
        """)
    
    with subtab_nasdaq:
        st.plotly_chart(
            generate_graphs._plot_index(df_nasdaq, label_index='Nasdaq'),
            use_container_width=True
        )

        st.markdown("""
        Assim como na comparação com Dow Jones, o índice Nasdaq também não tem uma correlação clara com relação à evolução do preço do
        petróleo tipo Brent
        """)

        corr_nasdaq = df_nasdaq.corr()
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(corr_nasdaq).update_layout(title=f'Correlação Brent x Nasdaq'),
            use_container_width=True,
        )
        st.markdown(f"""
        E calculando a correlação entre preço Brent x Nasdaq temos um valor $p$ **{corr_nasdaq.loc['Brent', 'Nasdaq']:.4f}** que é 
        classificado como uma correlação fraca.
        """)

