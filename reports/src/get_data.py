import pandas as pd
import numpy as np
import config
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap

@st.cache_data
def _df_passos_magicos():
    df = pd.read_csv('../data/raw/PEDE_PASSOS_DATASET_FIAP.csv', sep=';')
    df_evasao = _generate_df_evasao(df)
    # Mesclar os DataFrames com base na coluna 'NOME'
    df = df.merge(df_evasao[['NOME', 'EVADIU', 'ULTIMO_ANO']], on='NOME', how='left')
    # removendo linha problematica
    df = df.loc[~(df['INDE_2020'] == 'D980')]
    # substituindo null estranho
    df.loc[(df['INDE_2021'] == '#NULO!'),['INDE_2021']] = np.nan

    return df

@st.cache_data
def _df_boruta_shap(df):
    shap.initjs()
    cols = ['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020',
            'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021',
            'INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022', 'IPV_2022', 'IAN_2022', 'CG_2022', 'CF_2022', 'CT_2022',]
    df = df[cols + ['EVADIU']]

    _input_missings = SimpleImputer(strategy="median")
    df.loc[:, cols] = pd.DataFrame(_input_missings.fit_transform(df.loc[:, cols]), columns=cols)
    df = df.dropna(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
       df[cols], df['EVADIU'], test_size=0.2, random_state=42, shuffle=False
    )

    rf_estimator = GradientBoostingClassifier(random_state=42)

    rf_estimator.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_estimator)
    shap_values = explainer.shap_values(X_train)
    return df, X_train, y_train, shap_values, explainer

@st.cache_data
def _df_corr(df):
    cols = ['INDE_2020',  'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020',
            'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021',
            'INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022', 'IPV_2022', 'IAN_2022', 'CG_2022', 'CF_2022', 'CT_2022',]
    df = df[cols + ['EVADIU']]
    return df.corr()

@st.cache_data
def _generate_df_evasao(df):
    indicacao_evasao = ['NOME', 'PONTO_VIRADA_2020', 'PONTO_VIRADA_2021', 'PONTO_VIRADA_2022']
    df_evasao = df[indicacao_evasao]
    df_evasao['EVADIU'] = df_evasao.apply(verifica_evadidos, axis=1)
    # Aplicar a função para cada linha do DataFrame
    df_evasao['ULTIMO_ANO'] = df_evasao.apply(ultimo_ano, axis=1)
    return df_evasao

# Verificando quais alunos evadiram
def verifica_evadidos(row):
    if pd.isna(row['PONTO_VIRADA_2022']):  # Se PONTO_VIRADA_2022 for NaN
        return True  # O aluno evadiu
    elif pd.isna(row['PONTO_VIRADA_2020']):  # Se PONTO_VIRADA_2020 for NaN
        return False  # O aluno não evadiu
    elif pd.isna(row['PONTO_VIRADA_2021']):  # Se PONTO_VIRADA_2021 for NaN
        return True  # O aluno evadiu
    else:
        return False  # O aluno não evadiu

# Função para identificar o último ano de ponto de virada antes de NaN
def ultimo_ano(row):
    if row['EVADIU']:  # Verifica se EVADIU é verdadeiro
        if row['PONTO_VIRADA_2020'] in ['Sim', 'Não'] and pd.isna(row['PONTO_VIRADA_2021']):
            return 2020
        elif row['PONTO_VIRADA_2021'] in ['Sim', 'Não'] and pd.isna(row['PONTO_VIRADA_2022']):
            return 2021
        elif row['PONTO_VIRADA_2022'] in ['Sim', 'Não']:
            return 2022
    return pd.NA