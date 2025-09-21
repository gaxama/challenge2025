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
df_dados_clientes = pd.read_csv(DATA_DIR / "dados_clientes.csv", sep=';')
df_clientes_desde = pd.read_csv(DATA_DIR / "clientes_desde.csv", sep=';')
df_contratacoes_ultimos_12_meses = pd.read_csv(DATA_DIR / "contratacoes_ultimos_12_meses.csv", sep=';')

# Tratamento dos dfs
df_dados_clientes_tratado = funcs.trata_dados_clientes(df_dados_clientes)
df_clientes_desde_tratado = funcs.trata_clientes_desde(df_clientes_desde)
df_contratacoes_ultimos_12_meses_tratado = funcs.trata_contratacoes_ultimos_12_meses(df_contratacoes_ultimos_12_meses)

# Agrupamento dois dfs
df_agrupado = funcs.une_dataframes(df_dados_clientes_tratado, df_clientes_desde_tratado, df_contratacoes_ultimos_12_meses_tratado)

# Exportação dos dados tratados
OUTPUT_DIR = BASE_DIR / "dados" / "dados_tratados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_dados_clientes_tratado.to_csv(OUTPUT_DIR / "dados_clientes_tratados.csv", index=False, sep=';')
df_clientes_desde_tratado.to_csv(OUTPUT_DIR / "clientes_desde_tratados.csv", index=False, sep=';')
df_contratacoes_ultimos_12_meses_tratado.to_csv(OUTPUT_DIR / "contratacoes_ultimos_12_meses_tratados.csv", index=False, sep=';')
df_agrupado.to_csv(OUTPUT_DIR / "dados_tratados_agrupados.csv", index=False, sep=';')