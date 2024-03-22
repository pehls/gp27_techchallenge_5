import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
from dateutil.relativedelta import relativedelta
import shap

import src.get_data as get_data 

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
    df, X_train, y_train, shap_values, explainer = get_data._df_boruta_shap(df)
    return shap.force_plot(
        explainer.expected_value, shap_values, X_train.iloc
    )

def plot_shap2(df):
    df, X_train, y_train, shap_values, explainer = get_data._df_boruta_shap(df)
    return shap.summary_plot(shap_values, X_train)
