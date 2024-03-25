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
    df_features = df_features.loc[df_features['Importance'] > 0]
    return _h_bar_plot(df_features, y_col='Importance', x_col='Features', 
                       title=f'Feature Importance<br><sup>Total de {len(df_features)} features com importância na previsão</sup>', 
                       height=900)

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