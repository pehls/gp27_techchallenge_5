import config
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import shap
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

import joblib
import os
import streamlit as st
import numpy as np

@st.cache_data
def _get_tree_importances(_predict_pipeline):
    model = _predict_pipeline['model']
    df_importances = pd.DataFrame([_predict_pipeline[:-1].get_feature_names_out(), model.feature_importances_], index=['Features','Importance']).T
    df_importances = df_importances.loc[df_importances.Importance > 0.0001].sort_values('Importance', ascending=False)
    return df_importances

@st.cache_data
def check_causality(data : pd.DataFrame, list_of_best_features : list, y_col : str, threshold : float = 0.05) -> pd.DataFrame:
    data = data.dropna(axis=0, thresh=0.7).dropna(axis=1)
    from statsmodels.tsa.stattools import grangercausalitytests

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = pd.DataFrame(imp_mean.fit_transform(data), columns=data.columns)

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    maxlag=1
    test = 'ssr_chi2test'

    def grangers_causation_matrix(data, variables, focus = None, test=test, verbose=False):   
        variables = list(set(variables))
        if (type(focus) != type(None)): 
            df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        else:
            df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=focus)
        for c in df.columns:
            for r in df.index:
                try:
                    test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                    p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                    if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                    min_p_value = np.min(p_values)
                except:
                    print(f'r = {r}, c = {c}')
                    min_p_value = 99
                df.loc[r, c] = min_p_value
        df.index = [var + '_y' for var in df.index]
        df = df.reset_index()
        df.columns = ['variable'] + [var + '_x' for var in variables]
        return df
    g_matrix = (grangers_causation_matrix(data.dropna(), variables = list_of_best_features + [y_col], focus=[y_col]))
    #display(g_matrix)
    #g_matrix.columns = ['variable'] + [x + '_x' for x in list_of_best_features + ['QT_VOLUME_SELL_THROUGH']]
    
    # g_matrix = g_matrix.loc[g_matrix['QT_VOLUME_SELL_THROUGH_x'] < threshold]
    # var_y = list(set([x.replace('_y','') for x in g_matrix.variable]))
    g_matrix = g_matrix.loc[g_matrix['variable']==f'{y_col}_y'].T.reset_index()
    g_matrix.columns = g_matrix.iloc[0]
    g_matrix = g_matrix[1:].reset_index(drop=True)
    g_matrix = g_matrix.loc[g_matrix[f'{y_col}_y'] < threshold]
    g_matrix.variable = [x.replace('_x','').replace('minmax_', '') for x in g_matrix.variable]
    g_matrix.columns = ['Variable','Sign.']
    g_matrix = g_matrix.sort_values(f'Sign.', ascending=True)
    # g_matrix.index = g_matrix['Variable']
    return g_matrix[['Variable','Sign.']]

@st.cache_resource
def _get_xgb_model(path='models/xgb_model.pkl'):
    return joblib.load(path)

@st.cache_resource
def _run_xgboost(df_final, path='models/xgb_model.pkl', predict=False, retrain=False, sampling=True):
    cols = list(set(df_final.columns) - set(['EVADIU','NOME','YEAR']))
    X,y = df_final[cols], df_final['EVADIU']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    if (sampling):
        under = RandomUnderSampler()
        X_train, y_train = under.fit_resample(X_train, y_train)
    if not(retrain):
        if (os.path.isfile(path)) and not(predict):
            predict_pipeline = _get_xgb_model(path)
            return {
            'pipeline':predict_pipeline
            , 'precision':str(round(precision_score(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
            , 'recall':str(round(recall_score(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
            , 'confusion_matrix':confusion_matrix(y_test, predict_pipeline.predict(X_test))
            , 'roc_auc_score':round(roc_auc_score(y_test, predict_pipeline.predict(X_test)),4)
            , 'predictions':list(predict_pipeline.predict(X_test))
            , 'y_true':list(y_test)
            , 'X':X_test
            }
        
        if (os.path.isfile(path)) and (predict):
            predict_pipeline = _get_xgb_model(path)
            return {
            'pipeline':predict_pipeline
            , 'precision':str(round(precision_score(df_final['EVADIU'], predict_pipeline.predict(df_final[cols]))*100,2))+"%"
            , 'recall':str(round(recall_score(df_final['EVADIU'], predict_pipeline.predict(df_final[cols]))*100,2))+"%"
            , 'confusion_matrix':confusion_matrix(df_final['EVADIU'], predict_pipeline.predict(df_final[cols]))
            , 'roc_auc_score':round(roc_auc_score(df_final['EVADIU'], predict_pipeline.predict(df_final[cols])),4)
            , 'predictions':predict_pipeline.predict(df_final[cols])
            , 'y_true':df_final['EVADIU']
            , 'X':df_final[cols]
            }
    numeric_features = list(set(X_train.columns) - set(['YEAR']))
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    predict_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", GradientBoostingClassifier(random_state=42))]
    )

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    predict_pipeline.fit(X_train, y_train)
    joblib.dump(predict_pipeline, path)
    return {
          'pipeline':predict_pipeline
        , 'precision':str(round(precision_score(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
        , 'recall':str(round(recall_score(y_test, predict_pipeline.predict(X_test))*100,2))+"%"
        , 'confusion_matrix':confusion_matrix(y_test, predict_pipeline.predict(X_test))
        , 'roc_auc_score':round(roc_auc_score(y_test, predict_pipeline.predict(X_test)),4)
        , 'predictions':predict_pipeline.predict(X_test)
        , 'y_true':y_test
        , 'X':X_test
        }

@st.cache_data
def _get_shapley_values(df):
    cols = list(set(df.columns) - set(['EVADIU', 'NOME']))
    shap.initjs()
    response = _run_xgboost(df, path='models/xgb_model_forfeatures.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
       df[cols], df['EVADIU'], test_size=0.2, random_state=42, shuffle=False
    )

    rf_estimator = response['pipeline']['model']

    explainer = shap.TreeExplainer(rf_estimator)
    shap_values = explainer.shap_values(X_train)
    return X_train, y_train, shap_values, explainer

@st.cache_data
def _get_shap(_response):
    explainer = shap.TreeExplainer(_response['pipeline']['model'])
    shap_values = explainer(_response['X'])
    df_data = pd.DataFrame(
        np.c_[shap_values.base_values, shap_values.data],
        columns = ['Shap_Base'] + list(_response['X'].columns)
    ).reset_index(names=['id'])

    values = df_data.iloc[:,2:].abs().mean(axis=0).sort_values().index
    df_data[values] = MinMaxScaler().fit_transform(df_data[values])
    df_data.drop('Shap_Base', axis=1, inplace=True)

    df_data = pd.melt(df_data, id_vars=['id'], value_vars=values, var_name='Feature', value_name='Feature value')

    shap_df = pd.DataFrame(
        np.c_[shap_values.base_values, shap_values.values],
        columns = ['Shap_Base'] + list(_response['X'].columns)
    ).reset_index(names=['id'])
    shap_df.drop('Shap_Base', axis=1, inplace=True)
    df_plot = pd.melt(shap_df, id_vars=['id'], value_vars=values, var_name='Feature', value_name='SHAP value (impact on model output)')

    df_plot = df_plot.merge(
        df_data, on=['id','Feature']
    )

    return df_plot

@st.cache_resource
def get_explainer(_response, df_new_data):
    explainer = ClassifierExplainer(_response['pipeline'], df_new_data, _response['pipeline'].predict(df_new_data))
    explainer.pos_label = 1
    return explainer