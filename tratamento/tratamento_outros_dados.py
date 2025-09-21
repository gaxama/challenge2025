from pathlib import Path

import pandas as pd

import sys
import os

# Definindo raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import do arquivo funcs.py como módulo
from src import funcs

# Definindo diretório de dados de entrada
DATA_DIR = BASE_DIR / "dados" / "dados_totvs"

# Importando base de dados
df_telemetria = pd.read_csv(DATA_DIR / "telemetria_1.csv")
for i in range(2, 12):
    proximo_df = pd.read_csv(f'{DATA_DIR}\\telemetria_{i}.csv')
    df_telemetria = pd.concat([df_telemetria, proximo_df], ignore_index=True)
df_mrr = pd.read_csv(DATA_DIR / "mrr.csv", sep=';')
df_historico = pd.read_csv(DATA_DIR / "historico.csv", sep=';')
df_nps_relacional = pd.read_csv(DATA_DIR / "nps_relacional.csv", sep=';')

# Tratamento dos dfs
df_telemetria_tratado = funcs.trata_telemetria(df_telemetria)
df_mrr_tratado = funcs.trata_mrr(df_mrr)
df_historico_tratado = funcs.trata_historico(df_historico)
df_nps_relacional_tratado = funcs.trata_nps_relacional(df_nps_relacional)

# Exportação dos dados tratados
OUTPUT_DIR = BASE_DIR / "dados" / "dados_tratados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_telemetria_tratado.to_csv(OUTPUT_DIR / "dados_telemetria_tratados.csv", index=False)
df_historico_tratado.to_csv(OUTPUT_DIR / "dados_historico_tratados.csv", index=False, sep=';')
df_mrr_tratado.to_csv(OUTPUT_DIR / "dados_mrr_tratados.csv", index=False, sep=';')
df_nps_relacional_tratado.to_csv(OUTPUT_DIR / "nps_relacional_tratados.csv", index=False, sep=';')