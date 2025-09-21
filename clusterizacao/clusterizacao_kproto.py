from pathlib import Path

import pandas as pd

import sys
import os

# Definindo raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import do arquivo funcs.py como módulo
from src.funcs import *

# Definindo diretório de dados de entrada
DATA_DIR = BASE_DIR / "dados" / "dados_tratados"

# Import do df
df = pd.read_csv(DATA_DIR / "dados_tratados_agrupados.csv", sep=';')
# Definindo datasets para treinamento
df_agrupado, df_agrupado_2, x = trata_df_final(df, normalizar = False)
# Definindo k_mak como 6
k_max = 6
lista_init, valores_k = define_k_e_init(k_max, huang=False)
# Separando colunas categoricas
categorical_idx = trata_colunas_categoricas(x)
# Clusterização
clusteriza_kproto_final(valores_k, lista_init, df_agrupado, df_agrupado_2, x, categorical_idx, graficos=False)
# renomeando colunas do df final
df_agrupado_2.columns = ['CD_CLIENTE', 'DS_PROD_MODA', 'DS_LIN_REC_MODA', 'CIDADE',
       'DS_CNAE', 'DS_SEGMENTO', 'DS_SUBSEGMENTO',
       'MARCA_TOTVS_MODA', 'MODAL_COMERC_MODA', 'PAIS',
       'PERIODICIDADE_MODA', 'UF', 'VL_TOTAL_CONTRATO_SOMA',
       'FAIXA_VLR_TOT_SOMA_CONTRATOS', 'DIAS_CLIENTE', 'QTD_CONTRATACOES_12M',
       'VLR_CONTRATACOES_12M', 'CLUSTER_3', 'CLUSTER_4', 'CLUSTER_5',
       'CLUSTER_6']
# Definindo diretório de dados de entrada
OUTPUT_DIR = BASE_DIR / "dados" / "dataset_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Exportando dados
df_agrupado_2.to_csv(OUTPUT_DIR / 'clientes_clusterizados.csv', index=False, sep=';')