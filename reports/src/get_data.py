import pandas as pd
import numpy as np
import io
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

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
def _generate_df_evasao(df):
    indicacao_evasao = ['NOME', 'PONTO_VIRADA_2020', 'PONTO_VIRADA_2021', 'PONTO_VIRADA_2022']
    df_evasao = df[indicacao_evasao]
    df_evasao['EVADIU'] = df_evasao.apply(verifica_evadidos, axis=1)
    # Aplicar a função para cada linha do DataFrame
    df_evasao['ULTIMO_ANO'] = df_evasao.apply(ultimo_ano, axis=1)
    return df_evasao

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
def _get_idades(df=_df_passos_magicos()):
    df_idade = df[['IDADE_ALUNO_2020','FASE_TURMA_2020','NIVEL_IDEAL_2021', 'NIVEL_IDEAL_2022']]
    df_idade.loc[:,['IDADE_ALUNO_2020']] = df_idade.loc[:,['IDADE_ALUNO_2020']].fillna(0).astype(int)
    df_idade.loc[:,['IDADE_ALUNO_2021']] = [int(x) + 1 if int(x)>0 else 0 for x in df_idade['IDADE_ALUNO_2020']]
    df_idade.loc[:,['IDADE_ALUNO_2022']] = [int(x) + 1 if int(x)>0 else 0 for x in df_idade['IDADE_ALUNO_2021']]

    df_2021 = df_idade.groupby(['NIVEL_IDEAL_2021']).agg({'IDADE_ALUNO_2021':('min','median','max')}).reset_index()
    df_2022 = df_idade.groupby(['NIVEL_IDEAL_2022']).agg({'IDADE_ALUNO_2022':('min','median','max')}).reset_index()

    df_2021.columns = ['NIVEL_IDEAL_2021','IDADE_MIN_2021','IDADE_MEDIAN_2021','IDADE_MAX_2021']
    df_2022.columns = ['NIVEL_IDEAL_2022','IDADE_MIN_2022','IDADE_MEDIAN_2022','IDADE_MAX_2022']

    df_idades = df[['NOME','IDADE_ALUNO_2020','NIVEL_IDEAL_2021', 'NIVEL_IDEAL_2022']]
    df_idades['IDADE_ALUNO_2020'] = [dict_idade.get(getattr(x, 'NIVEL_IDEAL_2021'), np.nan) - 1 if not(float(getattr(x, 'IDADE_ALUNO_2020'))>0) else getattr(x, 'IDADE_ALUNO_2020') for x in df_idades.itertuples()]
    df_idades['IDADE_ALUNO_2020'] = [dict_idade.get(getattr(x, 'NIVEL_IDEAL_2022'), np.nan) - 2 if not(float(getattr(x, 'IDADE_ALUNO_2020'))>0) else getattr(x, 'IDADE_ALUNO_2020') for x in df_idades.itertuples()]

    df_idades.loc[:,['IDADE_ALUNO_2021']] = [float(x) + 1 if float(x)>0 else np.nan for x in df_idades['IDADE_ALUNO_2020']]
    df_idades.loc[:,['IDADE_ALUNO_2022']] = [float(x) + 1 if float(x)>0 else np.nan for x in df_idades['IDADE_ALUNO_2021']]

    return df_2021, df_2022, df_idades

dict_idade = {
     'ALFA (2º e 3º ano)':8
    ,'ALFA  (2º e 3º ano)':8
    ,'ALFA  (2o e 3o ano)':8
    ,'Fase 1 (4º ano)':10
    ,'Nível 1 (4o ano)':10
    ,'Fase 2 (5º e 6º ano)':11
    ,'Nível 2 (5o e 6o ano)':11
    ,'Fase 3 (7º e 8º ano)':13
    ,'Nível 3 (7o e 8o ano)':13
    ,'Fase 4 (9º ano)':14
    ,'Nível 4 (9o ano)':14
    ,'Fase 5 (1º EM)':15
    ,'ERRO':15
    ,'Nível 5 (1o EM)':15
    ,'Fase 6 (2º EM)':16
    ,'Nível 6 (2o EM)':16
    ,'Fase 7 (3º EM)':17
    ,'Nível 7 (3o EM)':17
    ,'Nível 8 (Universitários)':18
    ,'Fase 8 (Universitários)':18
}

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

    # agregar idade ajustada
    _, _, df_idades = _get_idades()


    # separar dados por ano
    df_2020 = df.loc[df.YEAR=='2020'].merge(
    df_idades[['NOME','IDADE_ALUNO_2020']].rename(columns={'IDADE_ALUNO_2020':'IDADE'})
    , on=['NOME'],how='left'
    )
    df_2021 = df.loc[df.YEAR=='2021'].merge(
    df_idades[['NOME','IDADE_ALUNO_2021']].rename(columns={'IDADE_ALUNO_2021':'IDADE'})
    , on=['NOME'],how='left'
    )
    df_2022 = df.loc[df.YEAR=='2022'].merge(
    df_idades[['NOME','IDADE_ALUNO_2022']].rename(columns={'IDADE_ALUNO_2022':'IDADE'})
    , on=['NOME'],how='left'
    )
    base_columns = ['NOME', 'IDADE', 'IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP',
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
                            , 'IDADE':'IDADE_Y-1'
                            })\
            .merge(df_2021[base_columns + ['YEAR','EVADIU']]\
                .rename(columns={  'IAA':'IAA_Y'
                                    , 'IAN':'IAN_Y'
                                    , 'IDA':'IDA_Y'
                                    , 'IEG':'IEG_Y'
                                    , 'INDE':'INDE_Y'
                                    , 'IPP':'IPP_Y'
                                    , 'IPS':'IPS_Y'
                                    , 'IPV':'IPV_Y'
                                    , 'IDADE':'IDADE_Y'
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
                            , 'IDADE':'IDADE_Y-1'
                            })\
            .merge(df_2022[base_columns + ['YEAR','EVADIU']]\
                .rename(columns={  'IAA':'IAA_Y'
                                    , 'IAN':'IAN_Y'
                                    , 'IDA':'IDA_Y'
                                    , 'IEG':'IEG_Y'
                                    , 'INDE':'INDE_Y'
                                    , 'IPP':'IPP_Y'
                                    , 'IPS':'IPS_Y'
                                    , 'IPV':'IPV_Y'
                                    , 'IDADE':'IDADE_Y'
                                    })
                    , on='NOME', how='right')
    df = pd.concat([df_2020_2021, df_2021_2022])
    df['IDADE_Y-1'] = df['IDADE_Y-1'].ffill()
    df['IDADE_Y'] = df['IDADE_Y'].ffill()

    df_saresp = _get_saresp()
    df_saresp_m1 = _rename(df_saresp, posfixo='_Y-1')
    for col in ['YEAR','IDADE_Y-1']:
        df_saresp_m1[col] = df_saresp_m1[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m1
        , on=['YEAR','IDADE_Y-1']
        )

    df_saresp_m1 = _rename(df_saresp, posfixo='_Y')
    for col in ['YEAR','IDADE_Y']:
        df_saresp_m1[col] = df_saresp_m1[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m1
        , on=['YEAR','IDADE_Y']
        )

    cols = list(set(df.columns) - set(['NOME', 'EVADIU','YEAR']))
    return df[cols + ['EVADIU']], cols

@st.cache_data
def _get_saresp():
    return pd.read_csv('data/processed/saresp.csv')

def _rename(df, cols_not_rename=['YEAR'], posfixo='_Y-1'):
    for col in cols_not_rename:
        df[col] = df[col].astype(str)
    for col in list(set(df.columns) - set(cols_not_rename)):
        df = df.rename(columns={col:col+posfixo})
    return df

@st.cache_data
def _read_file(file):
    return pd.read_excel(file)

def _return_download_data(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    return st.download_button(
        label="Exportar xlsx",
        data=buffer,
        file_name="data_example_simulations.xlsx",
        mime="application/vnd.ms-excel"
    )

def _test_df_cols(df):
    cols_in_last_year =  ['IDADE_Y-1', 'IAA_Y-1', 'IAN_Y-1', 'IDA_Y-1', 'IEG_Y-1', 'INDE_Y-1', 'IPP_Y-1', 'IPS_Y-1', 'IPV_Y-1']
    cols_in_current_year =  ['IDADE_Y', 'IAA_Y', 'IAN_Y', 'IDA_Y', 'IEG_Y', 'INDE_Y', 'IPP_Y', 'IPS_Y', 'IPV_Y']
    cols_identity = ['YEAR','NOME']
    cols_diff = set(cols_in_current_year + cols_in_last_year + cols_identity).difference(df.columns)
    return {
          'cols_diff':cols_diff
        , 'status_ok':True if len(cols_diff)<1 else False
        , 'df':df
    }

def _load_new_data(file_path):
    df = pd.read_excel(file_path)
    cols_in =  ['IDADE', 'IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']
    if not(_test_df_cols(df)['status_ok']):
        return _test_df_cols(df)
    
    # add dados da saresp em Y-1
    df_saresp = _get_saresp()
    df_saresp_m1 = _rename(df_saresp, posfixo='_Y-1')
    for col in ['YEAR','IDADE_Y-1']:
        df_saresp_m1[col] = df_saresp_m1[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m1
        , on=['YEAR','IDADE_Y-1']
        )
    
    # saresp no ano corrente
    df_saresp_m = _rename(df_saresp, posfixo='_Y')
    for col in ['YEAR','IDADE_Y']:
        df_saresp_m[col] = df_saresp_m[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m
        , on=['YEAR','IDADE_Y']
        )
    return {
          'cols_diff':[]
        , 'status_ok':True
        , 'df':df
    }

def _test_df_cols(df):
    cols_in_last_year =  ['IDADE_Y-1', 'IAA_Y-1', 'IAN_Y-1', 'IDA_Y-1', 'IEG_Y-1', 'INDE_Y-1', 'IPP_Y-1', 'IPS_Y-1', 'IPV_Y-1']
    cols_in_current_year =  ['IDADE_Y', 'IAA_Y', 'IAN_Y', 'IDA_Y', 'IEG_Y', 'INDE_Y', 'IPP_Y', 'IPS_Y', 'IPV_Y']
    cols_identity = ['YEAR','NOME']
    cols_diff = set(cols_in_current_year + cols_in_last_year + cols_identity).difference(df.columns)
    return {
          'cols_diff':cols_diff
        , 'status_ok':True if len(cols_diff)<1 else False
        , 'df':df
    }

def _load_new_data(file_path):
    df = pd.read_excel(file_path)
    cols_in =  ['IDADE', 'IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']
    if not(_test_df_cols(df)['status_ok']):
        return _test_df_cols(df)
    
    # add dados da saresp em Y-1
    df_saresp = _get_saresp()
    df_saresp_m1 = _rename(df_saresp, posfixo='_Y-1')
    for col in ['YEAR','IDADE_Y-1']:
        df_saresp_m1[col] = df_saresp_m1[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m1
        , on=['YEAR','IDADE_Y-1']
        )
    
    # saresp no ano corrente
    df_saresp_m = _rename(df_saresp, posfixo='_Y')
    for col in ['YEAR','IDADE_Y']:
        df_saresp_m[col] = df_saresp_m[col].astype(int)
        df[col] = df[col].astype(int)
    df = df.merge(
        df_saresp_m
        , on=['YEAR','IDADE_Y']
        )
    df.index = df['NOME']
    df = df.drop(columns={'NOME'})
    return {
          'cols_diff':[]
        , 'status_ok':True
        , 'df':df
    }
