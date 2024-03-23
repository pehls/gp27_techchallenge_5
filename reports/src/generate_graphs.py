import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import shap

from src import train_model, get_data

def plot_corr(df):
    fig = px.imshow(
          get_data._df_corr(df)
        , text_auto=True
        , aspect='auto',)
    fig.update_layout(
        height=1800
        ,width=1800
    )
    return fig

def plot_shap1(df):
    X_train, y_train, shap_values, explainer = train_model._get_shapley_values(df)
    return shap.force_plot(
        explainer.expected_value, shap_values, X_train.iloc
    )

def plot_shap2(df):
    X_train, y_train, shap_values, explainer = train_model._get_shapley_values(df)
    return shap.summary_plot(shap_values, X_train)

def _line_plot(df, x_col, y_col, title=''):
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

def _bar_plot(df, x_col, y_col, title=''):
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

def _h_bar_plot(df, x_col, y_col, title=''):
    fig = px.bar(
        df
        , x=x_col, y=y_col, orientation='h'
        , title=title
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

plot_style = {
      'Linhas' : _line_plot
    , 'Barras' : _bar_plot
    , "Barras Horizontais" : _h_bar_plot
}