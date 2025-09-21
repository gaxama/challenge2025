from pathlib import Path

import pandas as pd

import sys
import os

# Definindo raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Definindo diretório de dados de entrada
DATA_DIR_1 = BASE_DIR / "dados" / "dados_tratados"
DATA_DIR_2 = BASE_DIR / "dados" / "dataset_final"

# Imports
df = pd.read_csv(DATA_DIR_2 / "clientes_clusterizados.csv", sep=';')
df_telemetria_tratado = pd.read_csv(DATA_DIR_1 / "dados_telemetria_tratados.csv")
df_historico_tratado = pd.read_csv(DATA_DIR_1 / "dados_historico_tratados.csv", sep=';')
df_mrr_tratado = pd.read_csv(DATA_DIR_1 / "dados_mrr_tratados.csv", sep=';')
df_nps_relacional_tratado = pd.read_csv(DATA_DIR_1 / "nps_relacional_tratados.csv", sep=';')

# Merges
df = pd.merge(
    df, 
    df_telemetria_tratado, 
    on="CD_CLIENTE",
    how="left"
)
df = pd.merge(
    df, 
    df_historico_tratado, 
    on="CD_CLIENTE",
    how="left"
)
df = pd.merge(
    df, 
    df_mrr_tratado, 
    on="CD_CLIENTE",
    how="left"
)
df = pd.merge(
    df, 
    df_nps_relacional_tratado, 
    on="CD_CLIENTE",
    how="left"
)

# Arredondando valores p/ visualização
cols_float = df.select_dtypes(include='float').columns
df[cols_float] = df[cols_float].round(2)

# Export
df.to_csv(DATA_DIR_2 / "clientes_clusterizados_completo.csv", index=False, sep=';')