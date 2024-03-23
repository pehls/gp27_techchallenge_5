import pandas as pd
import numpy as np
import config
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def _df_passos_magicos():
    df = pd.read_csv('data/raw/PEDE_PASSOS_DATASET_FIAP.csv', sep=';')
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
    cols = ['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020',
            'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021',
            'INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022', 'IPV_2022', 'IAN_2022', 'CG_2022', 'CF_2022', 'CT_2022',]
    df = df[cols + ['EVADIU']]
    
    return df

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

@st.cache_data
def _get_modelling_data(df):
    # normalizar
    # scaler = MinMaxScaler()

    cols = ['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020',
            'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021',
            'INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022', 'IPV_2022', 'IAN_2022',]
    
    df = df[cols + ['EVADIU','ULTIMO_ANO','NOME']]
    # df[cols] = scaler.fit_transform(df[cols])

    df = df.melt(id_vars=['EVADIU','ULTIMO_ANO','NOME'])

    df['var_year'] = [x.split('_')[1] for x in df['variable']]
    df['variable'] = [x.split('_')[0] for x in df['variable']]
    df['EVADIU'] = [(str(getattr(x, 'ULTIMO_ANO')) == str(getattr(x, 'var_year'))) if not(getattr(x, 'ULTIMO_ANO') is pd.NA) else False for x in df.itertuples()]
    df = df.drop(columns=['ULTIMO_ANO'])
    df['value'] = df.value.astype(float)

    # pivotar para ficar indicador, ano, nome e se evadiu
    df = df.pivot_table(columns=['variable'], index=['EVADIU','var_year','NOME']).reset_index()
    cols = ['YEAR','IAA','IAN','IDA','IEG','INDE','IPP','IPS','IPV']
    df.columns = ['EVADIU','YEAR','NOME','IAA','IAN','IDA','IEG','INDE','IPP','IPS','IPV']

    # separar dados por ano
    df_2020 = df.loc[df.YEAR=='2020']
    df_2021 = df.loc[df.YEAR=='2021']
    df_2022 = df.loc[df.YEAR=='2022']
    base_columns = ['NOME', 'IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP',
        'IPS', 'IPV']
    
    # modificar para ficar Y-1 e Y
    df_2020_2021 = df_2020[base_columns]\
            .rename(columns={ 'IAA':'IAA_Y-1'
                            , 'IAN':'IAN_Y-1'
                            , 'IDA':'IDA_Y-1'
                            , 'IEG':'IEG_Y-1'
                            , 'INDE':'INDE_Y-1'
                            , 'IPP':'IPP_Y-1'
                            , 'IPS':'IPS_Y-1'
                            , 'IPV':'IPV_Y-1'
                            })\
            .merge(df_2021[base_columns + ['EVADIU']]\
                .rename(columns={  'IAA':'IAA_Y'
                                    , 'IAN':'IAN_Y'
                                    , 'IDA':'IDA_Y'
                                    , 'IEG':'IEG_Y'
                                    , 'INDE':'INDE_Y'
                                    , 'IPP':'IPP_Y'
                                    , 'IPS':'IPS_Y'
                                    , 'IPV':'IPV_Y'
                                    })
                    , on='NOME', how='right')

    df_2021_2022 = df_2021[base_columns]\
            .rename(columns={ 'IAA':'IAA_Y-1'
                            , 'IAN':'IAN_Y-1'
                            , 'IDA':'IDA_Y-1'
                            , 'IEG':'IEG_Y-1'
                            , 'INDE':'INDE_Y-1'
                            , 'IPP':'IPP_Y-1'
                            , 'IPS':'IPS_Y-1'
                            , 'IPV':'IPV_Y-1'
                            })\
            .merge(df_2022[base_columns + ['EVADIU']]\
                .rename(columns={  'IAA':'IAA_Y'
                                    , 'IAN':'IAN_Y'
                                    , 'IDA':'IDA_Y'
                                    , 'IEG':'IEG_Y'
                                    , 'INDE':'INDE_Y'
                                    , 'IPP':'IPP_Y'
                                    , 'IPS':'IPS_Y'
                                    , 'IPV':'IPV_Y'
                                    })
                    , on='NOME', how='right')
    df = pd.concat([df_2020_2021, df_2021_2022])

    cols = list(set(df.columns) - set(['NOME', 'EVADIU']))
    return df, cols

@st.cache_data
def _read_file(file):
    return pd.read_excel(file)