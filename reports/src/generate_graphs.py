import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import shap
import matplotlib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from explainerdashboard.custom import (
    ExplainerComponent, dbc, html,
    ShapDependenceComponent,
    ShapContributionsGraphComponent,
    ShapInteractionsComposite,
    ClassifierPredictionSummaryComponent,
    PdpComponent,
    ClassifierRandomIndexComponent,
    PosLabelSelector,
    IndexConnector, 
    to_html
)
from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard import ExplainerDashboard


from src import train_model, get_data

def plot_corr(df, show_only=None):
    if (show_only):
        data = get_data._df_corr(df)[show_only].sort_values(show_only)
    else:
        data = get_data._df_corr(df).sort_values(show_only)
    fig = px.imshow(
          data
        , text_auto=True
        , aspect='auto',)
    # fig.update_layout(
    #     height=1800
    #     ,width=1800
    # )
    return fig

def plot_shap_force_plot(df):
    shap.initjs()
    X_train, y_train, shap_values, explainer = train_model._get_shapley_values(df)
    shap_force_plot = shap.force_plot(
        explainer.expected_value, shap_values, X_train
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"
    return shap_html

def plot_shap_summary_plot(df):
    shap.initjs()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    X_train, y_train, shap_values, explainer = train_model._get_shapley_values(df)
    shap_summary_plot =  shap.summary_plot(shap_values, X_train)
    return fig

operations = {
      'Soma':'sum'
    , 'Média':'mean'
    , 'Mediana':'median'
}

def _line_plot(df, x_col, y_col, operation = None, title=''):

    if (operation):
        df[y_col] = df[y_col].astype(float, errors='ignore')
        df = df.groupby(x_col).agg({y_col : operations[operation]}).reset_index().sort_values(y_col)

    fig = go.Figure()

    fig = px.line(
        df, 
        x=x_col, y=y_col
        #, custom_data=['year', config.DICT_Y[stat][0], 'country_code']
    )

    # hide axes
    fig.update_xaxes(visible=True, title=title)
    fig.update_yaxes(visible=True,
                    gridcolor='black',zeroline=True,
                    showticklabels=True, title=''
                    )
    fig.update_layout(
        hovermode='x unified',
    )

    # fig.update_traces(
    #     hovertemplate="""
    #     <b>Metr.:</b> %{customdata[1]} 
    #     <b>Med.:</b> %{customdata[2]} 
    #     <b>Mediana:</b> %{customdata[3]} 
    #     <b>Max.:</b> %{customdata[4]} 
    #     """
    # )

    # strip down the rest of the plot
    fig.update_layout(
        showlegend=True,
        # plot_bgcolor="black",
        margin=dict(l=10,b=10,r=10)
    )
    return fig

def _bar_plot(df, x_col, y_col, operation = None, title=''):
    if (operation):
        df[y_col] = df[y_col].astype(float, errors='ignore')
        df = df.groupby(x_col).agg({y_col : operations[operation]}).reset_index().sort_values(y_col)
    fig = px.bar(
        df,
        x=x_col, y=y_col,
        title=title, text_auto=True
    )
    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False
        ),
        xaxis=dict(
            title=x_col,
            showgrid=False,
            showline=False,
            showticklabels=True
        )
    )
    fig.for_each_trace(lambda t: t.update(texttemplate = t.texttemplate + ''))

    return fig

def _h_bar_plot(df, x_col, y_col, operation = None, title='', height=None):
    _height=600
    if (operation):
        df[y_col] = df[y_col].astype(float, errors='ignore')
        df = df.groupby(x_col).agg({y_col : operations[operation]}).reset_index().reset_index().sort_values(y_col)
    if (height):
        _height=height
    fig = px.bar(
        df
        , x=y_col, y=x_col, orientation='h'
        , title=title, text=y_col, height=_height
    )
    fig.update_layout(
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True
        )
    )
    return fig

def _histogram_plot(df, x_col, title='Histograma', text_auto=True):
    fig = px.histogram(df, x=x_col, title=title, text_auto=text_auto)
    fig.update_layout(
        yaxis=dict(
            showgrid=True,
            showline=False,
            showticklabels=True
        ),
        xaxis=dict(
            showgrid=True,
            showline=False,
            showticklabels=True
        )
    )
    return fig

plot_style = {
      'Linhas' : _line_plot
    , 'Barras' : _bar_plot
    , "Barras Horizontais" : _h_bar_plot
    , 'Histograma': _histogram_plot
}

def _plot_confusion_matrix(cm):
    # create the heatmap
    heatmap = go.Heatmap(z=cm, x=['Not Churned', 'Churned'], y=['Not Churned', 'Churned'], colorscale='Blues')

    # create the layout
    layout = go.Layout(title='Confusion Matrix')

    # create the figure
    fig = go.Figure(data=[heatmap], layout=layout)
    return fig

def _plot_importance(df):
    pipeline = train_model._run_xgboost(df, path='models/xgb_model_forfeatures.pkl')['pipeline']
    feature_names = [x.split('__')[1] for x in pipeline[:-1].get_feature_names_out()]
    df_features = pd.DataFrame(columns=['Importance'], index=feature_names, data=pipeline['model'].feature_importances_).reset_index(names=['Features']).sort_values('Importance')
    df_features['Importance'] = [round(x, 2) for x in df_features['Importance']]
    df_features = df_features.loc[df_features['Importance'] > 0].sort_values(['Importance'])
    return _h_bar_plot(df_features.tail(10), y_col='Importance', x_col='Features', 
                       title=f'Feature Importance<br><sup>Total de {len(df_features)} features com importância na previsão</sup>')

def _plotly_plot_shap_values(response, width=800, height=600, show_only=15):
    df_plot = train_model._get_shap(response)
    df_order = df_plot
    df_order['abs_shap'] = abs(df_order['SHAP value (impact on model output)'])
    df_order = df_order.groupby('Feature').agg({'abs_shap':('sum','max')}).reset_index()
    df_order.columns = ['Feature','total_shap','max_shap']
    df_order = df_order.loc[df_order['total_shap']!=0].sort_values('max_shap', ascending=False)
    feat_out = list(set(response['X'].columns) - set(df_order.Feature.to_list()))

    fig = go.Figure()
    i = show_only
    y = list()
    for feature_name in df_order.head(show_only).Feature:
        internal_df = df_plot.loc[df_plot.Feature == feature_name]
        y_ = i + np.random.rand(len(df_plot))
        y.append(np.average(y_))
        fig.add_trace(go.Scatter(
            x=internal_df['SHAP value (impact on model output)'], 
            y=y_,
            mode='markers',
            marker=dict(
                size=9,
                color=internal_df['Feature value'],
                colorbar=dict(
                    title='Feature value',
                ),
                colorscale='Plasma'
            ),
            text=internal_df['Feature'],
            name=feature_name,
            customdata=round(internal_df['Feature value']*100,2),
            hovertemplate="<b>%{text}<br>Feature value: %{customdata}%<br>Impact on model out: %{x}<br>"
        ))
        i-=1

    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro', title='SHAP value (impact on model output)'),
        yaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro', title="Feature Value"),
        plot_bgcolor='white',
        title=f'''SHAP Values, Feature Values<br>
<sup>{len(df_order) - show_only} features com impacto além das {show_only} exibidas</sup><br>
<sup>{len(feat_out)} features sem impacto</sup>''',
        
    )
    fig.update_layout(plot_bgcolor='white', boxgap=0,
                    showlegend=False, coloraxis_showscale=True)
    fig.update_yaxes(tickvals=y, ticktext=df_order.Feature.to_list())
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )
    # fig.update_layout(hovermode="y unified")
    return fig


def _expose_explainer_custom_dashboard(_response, df_new_data):
    class CustomDashboard(ExplainerComponent):
        def __init__(self, explainer, title="Previsões e Variáveis Decisórias", **kwargs):
            super().__init__(explainer, title="Previsões e Variáveis Decisórias")
            self.pos_label = True
            self.index = ClassifierRandomIndexComponent(explainer,
                                                        hide_title=True, hide_index=False,
                                                        hide_slider=True, hide_labels=True,
                                                        hide_pred_or_perc=True,
                                                        hide_selector=True, hide_button=False)
            self.dependence = ShapDependenceComponent(explainer,
                                hide_selector=True, hide_percentage=True, hide_index=True,  
                                description="""
                                    Este gráfico mostra a relação entre as variáveis e seu valor shap, 
                                    fazendo com que consigamos investigar o relacionamento entre o valor das features e 
                                    o impacto na previsão. Você pode checar se o modelo usa as features de acordo com o
                                    esperado, ou usar o mesmo para aprender as relações que o modelo aprendeu entre
                                    as features de entrada e a saída predita
                                """ ,
                                title="Dependência Shap", subtitle="Relação entre o valor da variável e o valor SHAP. Em cinza, o NOME selecionado!",             
                                cutoff=0.75, **kwargs)
            self.pdp = PdpComponent(explainer,
                                hide_selector=True, hide_cats=True, hide_index=True,
                                hide_depth=True, hide_sort=True,
                                description="""
                                O gráfico de dependência parcial (pdp) mostra como seria a previsão do modelo
                                se você alterar um recurso específico. O gráfico mostra uma amostra
                                de observações e como essas observações mudariam com as mudanças aplicadas
                                (linhas de grade). O efeito médio é mostrado em cinza. O efeito
                                de alterar o recurso para um único Nome é
                                mostrado em azul. Você pode ajustar quantas observações serão utilizadas para o
                                calculo, quantas linhas de grade mostrar e quantos pontos ao longo do
                                eixo x para calcular as previsões do modelo.
                                """,
                                title="Gráfico de Dependência Parcial",
                                subtitle="Como a previsão mudará se você mudar uma variável?",
                                **kwargs)
            self.ind_preds = ShapContributionsGraphComponent(explainer,
                                hide_selector=True, hide_cats=True, hide_index=True,
                                hide_depth=True, hide_sort=True,
                                description="""
                                Este gráfico mostra a contribuição que cada característica individual tem 
                                na previsão de uma observação específica. 
                                As contribuições (a partir da média da população) somam-se à previsão final.
                                Isso permite explicar exatamente como cada previsão individual foi construída 
                                a partir de todos os ingredientes individuais do modelo.
                                """,
                                title="Gráfico das Contribuições", subtitle="Como cada Variável contribuiu para a previsão?",
                                **kwargs)
            self.class_preds = ClassifierPredictionSummaryComponent(explainer,
                                hide_selector=True, hide_cats=True, hide_index=True,
                                hide_depth=True, hide_sort=True,
                                description="Mostra a probabilidade predita para cada situação, sendo False para não-evasão e True para Evasão.",
                                title="Previsões & Probabilidades",
                                **kwargs)
            self.connector = IndexConnector(self.index, [self.dependence, self.pdp, self.ind_preds, self.class_preds])

        def layout(self):
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Selecione um Aluno:"),
                        self.index.layout()
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        self.ind_preds.layout(),
                    ]),
                    dbc.Col([
                        self.class_preds.layout(),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.dependence.layout(),
                    ]),
                    dbc.Col([
                        self.pdp.layout(),
                    ]),
                ])
                
            ])
    import dash, flask
    app = dash.Dash(__name__)

    server =  flask.Flask(__name__)

    app.scripts.config.serve_locally=False
    app.css.config.serve_locally=False
    
    explainer = ClassifierExplainer(_response['pipeline'], df_new_data, _response['pipeline'].predict(df_new_data))
    exp_dash = ExplainerDashboard(explainer, 
                                CustomDashboard, 
                                server=server, url_base_pathname="/explainer_dashboard/",
                                header_hide_selector=True, 
                                description = "Esta área do dashboard mostra o funcionamento do modelo, explicando como ele realizou as suas predições")
    # exp_dash.run(8050, mode='inline')
    _serve_flask(exp_dash, app)
    return "https://0.0.0.0:5000/explainer_dashboard/"

def _serve_flask(exp_dash, app_dash):
    from werkzeug.serving import make_server
    import flask, threading, dash
    from flask import request
    import socket

    def shutdown_server():
        global server
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        server.shutdown()
    class ServerThread(threading.Thread):
        def __init__(self, app):
            threading.Thread.__init__(self)
            self.server = make_server('0.0.0.0', 5000, app)
            self.ctx = app.app_context()
            self.ctx.push()

        def run(self):
            self.server.serve_forever()

        def shutdown(self):
            self.server.shutdown()

    def start_server():
        global server
        app = flask.Flask('myapp')
        server = ServerThread(app)

        @app.route('/explainer_dashboard/')
        def return_dashboard():
            return exp_dash.app.index()
        
        @app.route('/quit')
        def _quit():
            shutdown_server()
        
        return app


    app = dash.Dash(server=start_server())
    try:
        shutdown_server()
        
    except:
        print("ok")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('0.0.0.0',5000)) != 0:
        app.run_server(port=5000, host='0.0.0.0')
    sock.close()