import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from yellowbrick.cluster import KElbowVisualizer

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Definindo funções de agregação: moda

def agg_moda(x):
    m = x.dropna().mode()
    if not m.empty:
        return m.iloc[0]
    return np.nan


# Definindo funções de agregação: moda com tratamento especial para faixa de faturamento

def agg_moda_faixa(x):
    contagem = x.dropna().value_counts()
    if contagem.empty:
        return np.nan
    mais_frequente = contagem.index[0]
    if mais_frequente == "Sem Informações de Faturamento" and len(contagem) > 1:
        return contagem.index[1]
    return mais_frequente


def trata_dados_clientes(df):
    # Convertendo valores monetarios para floar/int
    df["VL_TOTAL_CONTRATO"] = df["VL_TOTAL_CONTRATO"].str.replace(",", ".").astype(float)
    # convertendo data para valor inteiro
    df["DT_ASSINATURA_CONTRATO"] = df["DT_ASSINATURA_CONTRATO"].str.replace("-", "").astype(int)
    # Transformando contratos de valor negativo em 0
    df.loc[df["VL_TOTAL_CONTRATO"] < 0, "VL_TOTAL_CONTRATO"] = 0
    # Preenchendo valores nulos de algumas colunas como "Não Informado"
    df['DS_SUBSEGMENTO'] = df['DS_SUBSEGMENTO'].fillna('NAO INFORMADO')
    df['MARCA_TOTVS'] = df['MARCA_TOTVS'].fillna('NAO INFORMADO')
    df['MODAL_COMERC'] = df['MODAL_COMERC'].fillna('NAO INFORMADO')
    df['SITUACAO_CONTRATO'] = df['SITUACAO_CONTRATO'].fillna('NAO INFORMADO')
    # Criando o dataframe agrupado
    df_agrupado = df.groupby('CD_CLIENTE').agg({'DS_PROD':agg_moda,
                             'DS_LIN_REC':agg_moda,
                             'CIDADE':agg_moda,
                             'DS_CNAE':agg_moda,
                             'DS_SEGMENTO':agg_moda,
                             'DS_SUBSEGMENTO':agg_moda,
                             'MARCA_TOTVS':agg_moda,
                             'MODAL_COMERC':agg_moda,
                             'PAIS':agg_moda,
                             'PERIODICIDADE':agg_moda,
                             'UF':agg_moda,
                             'VL_TOTAL_CONTRATO':'sum'}).reset_index()
    # Renomeando colunas
    df_agrupado.columns = ['CD_CLIENTE', 'DS_PROD_moda', 'DS_LIN_REC_moda', 'CIDADE_moda', 'DS_CNAE_moda',
       'DS_SEGMENTO_moda', 'DS_SUBSEGMENTO_moda', 'MARCA_TOTVS_moda',
       'MODAL_COMERC_moda', 'PAIS_moda', 'PERIODICIDADE_moda', 'UF_moda',
       'VL_TOTAL_CONTRATO_soma']
    # Dividindo os valores de contrato em faixas
    tamanho_faixa = 500_000
    max_valor = df_agrupado["VL_TOTAL_CONTRATO_soma"].max()
    limites = np.arange(0, max_valor + tamanho_faixa, tamanho_faixa)
    labels = [f"{int(l/500_000)*0.5}-{int(limites[i+1]/500_000)*0.5}M" for i, l in enumerate(limites[:-1])]
    df_agrupado["faixa_vl_total_soma_contratos"] = pd.cut(df_agrupado["VL_TOTAL_CONTRATO_soma"], bins=limites, labels=labels, include_lowest=True)
    # Convertendo codigo do pais para string
    df_agrupado['PAIS_moda'] = df_agrupado['PAIS_moda'].astype(str)
    return df_agrupado


def trata_clientes_desde(df_clientes_desde):
    df_clientes_desde["CLIENTE_DESDE"] = pd.to_datetime(
        df_clientes_desde["CLIENTE_DESDE"], format="%Y-%m-%d", errors="coerce"
    )
    df_clientes_desde["DIAS_CLIENTE"] = (
        pd.Timestamp.today().normalize() - df_clientes_desde["CLIENTE_DESDE"]
    ).dt.days.astype("Int64")
    df_clientes_desde = df_clientes_desde.drop("CLIENTE_DESDE", axis=1)
    df_clientes_desde.rename(columns={"CLIENTE": "CD_CLIENTE"}, inplace=True)
    return df_clientes_desde


def trata_contratacoes_ultimos_12_meses(df_contratacoes_ultimos_12_meses):
    df_contratacoes_ultimos_12_meses['VLR_CONTRATACOES_12M'] = df_contratacoes_ultimos_12_meses['VLR_CONTRATACOES_12M'].str.replace(",", ".").astype(float)
    df_contratacoes_ultimos_12_meses[["QTD_CONTRATACOES_12M", "VLR_CONTRATACOES_12M"]] = (
        df_contratacoes_ultimos_12_meses[["QTD_CONTRATACOES_12M", "VLR_CONTRATACOES_12M"]].fillna(0)
    )
    return df_contratacoes_ultimos_12_meses


def trata_nps_relacional(df):
    df = df.drop(['respondedAt'], axis=1)
    colunas_para_preencher = df.columns.tolist()
    colunas_para_preencher.remove("resposta_NPS")
    colunas_para_preencher.remove("metadata_codcliente")
    coluna_base = "resposta_NPS"
    for col in colunas_para_preencher:
        df[col] = df[col].fillna(df[coluna_base])
    df_agregado = df.groupby('metadata_codcliente').agg('mean').reset_index()
    df_agregado = df_agregado.rename(columns={'metadata_codcliente': 'CD_CLIENTE'})
    return df_agregado
    

def trata_telemetria(df_telemetria):
    # Excluindo colunas indesejadas
    df_telemetria = df_telemetria.drop(['tcloud', 'clienteprime', 'referencedatestart'], axis=1)
    # Conmvetendo o código da loja em código do cliente (removendo últiumos 2 dígitos)
    df_telemetria['clienteid'] = df_telemetria['clienteid'].str[:-2]
    # Criando coluna para contagem de casos
    df_telemetria['contagem'] = 1
    # Agrupando o df por cliente
    df_telemetria_agrupado = df_telemetria.groupby('clienteid').agg({
                                'moduloid': agg_moda,
                                'productlineid': agg_moda,
                                'slotid': agg_moda,
                                'statuslicenca': agg_moda,
                                'contagem': 'count'}).reset_index()
    df_telemetria_agrupado.columns = ['CD_CLIENTE',
                       'telemetria_moduloid_moda',
                       'telemetria_productlineid_moda',
                       'telemetria_slotid_moda',
                       'telemetria_statuslicenca_moda',
                       'telemetria_contagem']
    return df_telemetria_agrupado


def trata_historico(df):
    # Trocando ',' por '.'
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # Convertendo coluna numericas para float
    cols_para_float = ['ITEM_PROPOSTA',
        'QTD', 'VL_PCT_DESC_TEMP',
        'VL_PCT_DESCONTO', 'PRC_UNITARIO', 'VL_DESCONTO_TEMPORARIO', 'VL_TOTAL',
        'VL_FULL', 'VL_DESCONTO']
    df[cols_para_float] = df[cols_para_float].astype(float)
    # Renomeando coluna cliente
    df = df.rename(columns={"CD_CLI": "CD_CLIENTE"})
    # Agrupando
    df_agrupado = df.groupby('CD_CLIENTE').agg({'NR_PROPOSTA':'count',
                                            'ITEM_PROPOSTA': agg_moda,
                                            'HOSPEDAGEM': agg_moda,
                                            'FAT_FAIXA': agg_moda,
                                            'CD_PROD': agg_moda,
                                            'QTD': ['median', 'sum'],
                                            'VL_PCT_DESC_TEMP': ['median', 'sum'],
                                            'VL_PCT_DESCONTO': ['median', 'sum'],
                                            'PRC_UNITARIO': ['median', 'sum'],
                                            'VL_DESCONTO_TEMPORARIO': ['median', 'sum'],
                                            'VL_TOTAL': ['median', 'sum'],
                                            'VL_FULL': ['median', 'sum'],
                                            'VL_DESCONTO': ['median', 'sum']}).reset_index()
    df_agrupado.columns = [
        'CD_CLIENTE',
        'NR_PROPOSTA_count',
        'ITEM_PROPOSTA_agg_moda',
        'HOSPEDAGEM_agg_moda',
        'FAT_FAIXA_agg_moda',
        'CD_PROD_agg_moda',
        'QTD_median',
        'QTD_sum',
        'VL_PCT_DESC_TEMP_median',
        'VL_PCT_DESC_TEMP_sum',
        'VL_PCT_DESCONTO_median',
        'VL_PCT_DESCONTO_sum',
        'PRC_UNITARIO_median',
        'PRC_UNITARIO_sum',
        'VL_DESCONTO_TEMPORARIO_median',
        'VL_DESCONTO_TEMPORARIO_sum',
        'VL_TOTAL_median',
        'VL_TOTAL_sum',
        'VL_FULL_median',
        'VL_FULL_sum',
        'VL_DESCONTO_median',
        'VL_DESCONTO_sum'
    ]
    return df_agrupado


def trata_mrr(df):
    df = df.rename(columns={"CLIENTE": "CD_CLIENTE"})
    df_agrupado = df.groupby('CD_CLIENTE').agg('median').reset_index()
    df_agrupado = df_agrupado.rename(columns={"MRR_12M": "MRR_12M_median"})
    return df_agrupado



def une_dataframes(df_dados_clientes_tratado, df_clientes_desde_tratado, df_contratacoes_ultimos_12_meses_tratado):
    df_agrupado = pd.merge(
        df_dados_clientes_tratado, 
        df_clientes_desde_tratado, 
        on="CD_CLIENTE",
        how="left"
    )
    df_agrupado = pd.merge(
        df_agrupado, 
        df_contratacoes_ultimos_12_meses_tratado, 
        on="CD_CLIENTE",
        how="left"
    )
    df_agrupado[["QTD_CONTRATACOES_12M", "VLR_CONTRATACOES_12M"]] = (
        df_agrupado[["QTD_CONTRATACOES_12M", "VLR_CONTRATACOES_12M"]].fillna(0)
    )
    return df_agrupado


# Tratamento final do df para clusterização

def trata_df_final(df_agrupado, normalizar = False):
    # Criando df para controle
    df_agrupado_2 = df_agrupado.copy(deep=True)
    # Normalizar colunas numéricas do df original (padrão é não)
    if normalizar:
        for coluna in df_agrupado.select_dtypes(include=['float64', 'int64']).columns:
            if coluna != 'CD_CLIENTE':
                df_agrupado[[coluna]] = MinMaxScaler().fit_transform(df_agrupado[[coluna]])
    # Criando o dataset para alimentar o modelo
    x = df_agrupado.copy(deep=True)
    # Transformando a coluna CD_CLIENTE em numérica para usar no KPrototypes
    x['CD_CLIENTE'], id_map = pd.factorize(x['CD_CLIENTE'])
    return df_agrupado, df_agrupado_2, x


# Função para exibir as diferentes avaliações para o número de clusters

def define_numero_clusters(df_agrupado_3):
    features = df_agrupado_3.columns.tolist()
    features.remove('CD_CLIENTE')
    x = df_agrupado_3[features].values
    # Elbow method
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,30))
    visualizer.fit(x)
    visualizer.show()
    # Silhouette method
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,30), metric='silhouette', timings=True)
    visualizer.fit(x)
    visualizer.show()
    # Calinski harabaz method
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,30), metric='calinski_harabasz', timings=True)
    visualizer.fit(x)
    visualizer.show()


# Função para visualização da distribuiçlão dos valores de cada coluna por cluster

def visualiza_colunas_por_cluster(df, k):
    colunas_graficos = ['faixa_vl_total_soma_contratos', 'DS_SEGMENTO_moda', 'UF_moda']
    colunas_histograma = ['DIAS_CLIENTE', 'VLR_CONTRATACOES_12M']  # ajuste conforme necessário
    # Gráficos de barras empilhados (1x3)
    fig_bar, axes_bar = plt.subplots(len(colunas_graficos), 1, figsize=(15, 5 * len(colunas_graficos)))
    if len(colunas_graficos) == 1:
        axes_bar = [axes_bar]
    for i, coluna in enumerate(colunas_graficos):
        contagem = pd.crosstab(df[coluna], df[f"cluster_{k}"])
        contagem.plot(kind="bar", stacked=True, colormap="Set2", ax=axes_bar[i])
        axes_bar[i].set_xlabel(coluna)
        axes_bar[i].set_ylabel("Contagem")
        axes_bar[i].set_title(f"Contagem de {coluna} por cluster para k={k}")
        axes_bar[i].legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    # Histogramas empilhados (1x2)
    fig_hist, axes_hist = plt.subplots(len(colunas_histograma), 1, figsize=(15, 5 * len(colunas_histograma)))
    if len(colunas_histograma) == 1:
        axes_hist = [axes_hist]
    # Escolha o colormap
    cmap = cm.get_cmap('Set2', len(df[f'cluster_{k}'].unique()))
    colors = [cmap(i) for i in range(len(df[f'cluster_{k}'].unique()))]

    for j, coluna in enumerate(colunas_histograma):
        for idx, cluster in enumerate(sorted(df[f'cluster_{k}'].unique())):
            subset = df[df[f'cluster_{k}'] == cluster][coluna]
            axes_hist[j].hist(subset, bins=50, alpha=0.8, label=f'Cluster {cluster}', color=colors[idx])
        axes_hist[j].set_title(f'Histograma de {coluna} por cluster para k={k}')
        axes_hist[j].set_xlabel(coluna)
        axes_hist[j].set_ylabel('Frequência')
        axes_hist[j].legend()
    plt.tight_layout()
    plt.show()


# Função para exibir a contagem do número de clientes e valor total dos contratos

def mostra_contagem_clientes_casos(df, k):
    df_contagem_clientes_casos = df.groupby(f'cluster_{k}').agg({
        'CD_CLIENTE': 'count',
        'VL_TOTAL_CONTRATO_soma': ['sum', 'mean']
    }).reset_index()
    df_contagem_clientes_casos.columns = [f'cluster_{k}', 'numero_clientes', 'valor_total_contratos', 'media_total_contratos']
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df_contagem_clientes_casos)


# Função para clusterazação com kprototype

def clusterização_kproto(x, k, init, categorical, max_iter=100):
    kproto = KPrototypes(n_clusters=k, init=init, max_iter=max_iter, verbose=1, random_state=42)
    kproto.fit(x, categorical=categorical)
    labels = kproto.labels_
    cluster_pred = kproto.predict(x, categorical=categorical)

    return kproto, labels, cluster_pred


# Função para avaliadção da clusterização do kprototype - a avaliação exige apenas valores numéricos

def avaliacao_clusterizacao_kproto(x, labels):
    # Cria uma cópia para não alterar o original
    x_temp = x.drop('CD_CLIENTE', axis=1).copy()
    # Identifica colunas categóricas (object ou category)
    colunas_categoricas = x_temp.select_dtypes(include=['object', 'category']).columns
    # Aplica frequency encoding nas colunas categóricas
    for coluna in colunas_categoricas:
        freq = x_temp[coluna].value_counts(normalize=True)
        x_temp[coluna] = x_temp[coluna].map(freq)
    # Calcula as métricas usando apenas valores numéricos
    x_numerico = x_temp.select_dtypes(include=[np.number])
    score = silhouette_score(x_numerico, labels)
    db_index = davies_bouldin_score(x_numerico, labels)
    ch_index = calinski_harabasz_score(x_numerico, labels)
    return score, db_index, ch_index


# Função para alimentar dataframe da avaliação da clusterização

def alimenta_df_avaliacao_clusterizacao_kproto(df_avaliacao, k, init, score, db_index, ch_index):
    df_avaliacao.loc[len(df_avaliacao)] = [k, init, score, db_index, ch_index]
    return df_avaliacao


# Definindo limite de k e parametros para teste

def define_k_e_init(k_max, huang=False):
    valores_k = list(range(3, (k_max+1)))
    if huang == False:
        lista_init = ["Cao"]
    else:
        lista_init = ["Huang", "Cao"]
    return lista_init, valores_k


# Transformando colunas de objetos em categoricas

def trata_colunas_categoricas(x):
    for col in x.select_dtypes(include=['object']).columns:
        x[col] = x[col].astype('category')
    # Separando colunas categoricas
    categorical_idx = [x.columns.get_loc(col) for col in x.select_dtypes(include=['category']).columns]
    return categorical_idx


# Pipeline de clusterização com diversos valores de k

def clusteriza_kproto_final(valores_k, lista_init, df_agrupado, df_agrupado_2, x, categorical_idx, graficos=False):
    # Lista com os nomes das colunas de avaliação
    colunas_avaliacao = ["clusters", "init", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]

    # Cria um DataFrame vazio para avaliação
    df_avaliacao_2 = pd.DataFrame(columns=colunas_avaliacao)

    best_score_overall = 0
    best_k_overall = 0

    for k in valores_k:
        best_score = 0
        best_labels = None
        print(f"\nClusterização com k={k}:\n")
        for init in lista_init:
            # Clusterização
            print(f"Clusterização com k={k}, init={init}")
            kproto, labels, cluster_pred = clusterização_kproto(x, k, init, categorical_idx)

            # Alimentando o df de avaliação geral dos clusters
            score, db_index, ch_index = avaliacao_clusterizacao_kproto(x, labels)
            print(f"Silhouette Score: {score:.4f}, Davies-Bouldin Index: {db_index:.4f}, Calinski-Harabasz Index: {ch_index:.4f}\n")
            alimenta_df_avaliacao_clusterizacao_kproto(df_avaliacao_2, k, init, score, db_index, ch_index)

            # Usando o sihlouette score para atribuir o melhor modelo dentro de cada k
            if best_score == 0:
                best_score = score
                best_labels = labels
            elif score > best_score:
                best_score = score
                best_labels = labels

        # Alimentando os dfs originais com os clusters 
        df_agrupado[f"cluster_{k}"] = best_labels
        df_agrupado_2[f"cluster_{k}"] = best_labels

        # Exibição no número de clientes e casos por cluster
        print(f"\nNúmero de clientes ,valor total e média dos contratos totais por cluster para k={k}:\n")
        mostra_contagem_clientes_casos(df_agrupado, k)

        if graficos == True:
            # Gerando os gráficos de distribuiução dos valores de cada coluna por cluster
            print(f"\nGráficos de distribuição para k={k}:\n")
            visualiza_colunas_por_cluster(df_agrupado, k)

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_k_overall = k

    # Exibindo df com a avaliação da clusterização
    print("\nComparação final entre os indicadores de cada clusterização:\n")
    print(f"\nMelhor k: {best_k_overall} com base no silhouette score ({best_score_overall:.4f}).\n")
    print(df_avaliacao_2)