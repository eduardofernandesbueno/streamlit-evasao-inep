# Script: analises.py
# Objetivo: Analisar dados de cursos e calcular/visualizar as taxas de conclusão e evasão por curso e ano

import os
import re

# ==== EDA/Audit Imports e Auditoria Inicial ====
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / "src"

# Garante que a raiz do projeto e a pasta src estejam no sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evasao.utils.inspecao import auditar_df, registrar_ambiente
from evasao.utils.paths import DADOS_PROCESSADO

# Diretório padrão onde os CSVs processados são gravados
PASTA_PROCESSADO = str(DADOS_PROCESSADO)

registrar_ambiente(etapa="analises", contexto="antes_eda")

# ---------------------------------------------------------------------
# Base descritiva / EDA geral
# ---------------------------------------------------------------------
# Esta base continua útil para:
# - EDA
# - estatísticas descritivas
# - gráficos exploratórios
# - tabelas de apoio
# Mas não é a base principal de treino do RF do artigo SBIE.
df_eda = pd.read_csv(
    os.path.join(PASTA_PROCESSADO, "dados_ingresso_evasao_conclusao.csv"),
    sep=";",
    encoding="utf-8",
    low_memory=False,
)

# Auditoria antes de qualquer transformação mais pesada
auditar_df(df_eda, etapa="analises", contexto="df_bruto_para_eda", n=5)

# ---------------------------------------------------------------------
# Base temporal SBIE
# ---------------------------------------------------------------------
# Base principal metodológica do artigo para modelagem preditiva temporal.
caminho_temporal_sbie = os.path.join(
    PASTA_PROCESSADO, "dados_modelagem_temporal_sbie.csv"
)
if os.path.exists(caminho_temporal_sbie):
    df_temporal_sbie = pd.read_csv(
        caminho_temporal_sbie,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )
    auditar_df(
        df_temporal_sbie,
        etapa="analises",
        contexto="df_temporal_sbie_para_eda",
        n=5,
    )
else:
    print(f"Arquivo temporal SBIE não encontrado: {caminho_temporal_sbie}")
    df_temporal_sbie = pd.DataFrame()

# Define o mapeamento de cores: verde para "Taxa de Conclusão" e vermelho para "Taxa de Evasão"
color_map = {"Taxa de Conclusão": "green", "Taxa de Evasão": "red"}

# =============================================================================
# Leitura dos dados de cursos para os anos de 2009 a 2024
# =============================================================================
anos = range(2009, 2025)
ANO_CORTE = 2018
lista_cursos: list[pd.DataFrame] = []

# Lista de colunas que devem ser numéricas (subconjunto curso_geral_compacto)
numeric_cols = [
    "numero_cursos",
    "vagas_totais",
    "ingressantes",
    "concluintes",
    "inscritos_totais",
]

for ano in anos:
    caminho = os.path.join(PASTA_PROCESSADO, f"dados_cursos_tratado_{ano}.csv")
    if os.path.exists(caminho):
        # Lê tudo inicialmente como string para depois converter colunas numéricas
        df = pd.read_csv(caminho, sep=";", dtype=str, encoding="utf-8")
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ano"] = ano
        lista_cursos.append(df)
    else:
        print(f"Arquivo não encontrado: {caminho}")

if not lista_cursos:
    raise FileNotFoundError(
        "Nenhum arquivo 'dados_cursos_tratado_*.csv' foi encontrado para os anos especificados."
    )

# Concatena todos os anos e aplica split temporal (coerente com o pré-processamento)
dados_cursos_tratado = pd.concat(lista_cursos, ignore_index=True)
df_treino = dados_cursos_tratado[dados_cursos_tratado["ano"] <= ANO_CORTE]
df_teste = dados_cursos_tratado[dados_cursos_tratado["ano"] > ANO_CORTE]

print("\n" + "=" * 70)
print("ANÁLISES DESCRITIVAS INICIADAS")
print("=" * 70)
print("Base descritiva carregada: dados_ingresso_evasao_conclusao.csv")
print(f"Linhas base EDA: {len(df_eda):,}".replace(",", "."))
print(f"Colunas base EDA: {len(df_eda.columns)}")

if not df_temporal_sbie.empty:
    print("Base temporal SBIE carregada: dados_modelagem_temporal_sbie.csv")
    print(f"Linhas base temporal: {len(df_temporal_sbie):,}".replace(",", "."))
    print(f"Colunas base temporal: {len(df_temporal_sbie.columns)}")
else:
    print("Base temporal SBIE não encontrada para apoio analítico.")

print(f"Arquivos anuais de cursos carregados: {len(lista_cursos)}")
print(
    "Checagem de consistência temporal (cursos tratados):",
    len(df_treino),
    len(df_teste),
    len(dados_cursos_tratado),
)
print(
    f"Janela temporal descritiva - treino até {ANO_CORTE}: {len(df_treino):,} linhas".replace(
        ",", "."
    )
)
print(
    f"Janela temporal descritiva - teste após {ANO_CORTE}: {len(df_teste):,} linhas".replace(
        ",", "."
    )
)
print(
    f"Janela temporal descritiva - total cursos tratados: {len(dados_cursos_tratado):,} linhas".replace(
        ",", "."
    )
)

print("\n[INFO] Este script é de apoio descritivo.")
print(
    "[INFO] O centro metodológico do SBIE permanece no randomforest.py com base temporal sem leakage."
)
print(
    "[INFO] As curvas por curso com duração curricular assumida são apenas visualizações descritivas."
)
print("[INFO] Não use essas curvas como evidência principal de performance preditiva.")
print("=" * 70 + "\n")

# =============================================================================
# Leitura dos dados de universidades (IES) – utilizando a base correspondente
# a cada ano, para manter a parte descritiva historicamente mais rigorosa
# =============================================================================
lista_ies: list[pd.DataFrame] = []

for ano in anos:
    ies_path = os.path.join(PASTA_PROCESSADO, f"dados_ies_tratado_{ano}.csv")
    if os.path.exists(ies_path):
        df_ies_ano = pd.read_csv(ies_path, sep=";", dtype=str, encoding="utf-8")
        df_ies_ano["ano"] = ano
        lista_ies.append(df_ies_ano)
    else:
        print(f"Arquivo de IES não encontrado: {ies_path}")

if not lista_ies:
    raise FileNotFoundError(
        "Nenhum arquivo 'dados_ies_tratado_*.csv' foi encontrado para os anos especificados."
    )

dados_ies_tratado = pd.concat(lista_ies, ignore_index=True)

# =============================================================================
# Unindo os dados de cursos (apenas período de treino) e universidades
# =============================================================================
all_data = pd.merge(
    df_treino,
    dados_ies_tratado,
    on=["id_ies", "ano"],
    how="left",
)

# Filtra para manter apenas IES com número de cursos válido e nome não nulo
all_data = all_data[
    (pd.to_numeric(all_data["numero_cursos"], errors="coerce") > 0)
    & (all_data["nome_ies"].notnull())
]

# Normaliza textos
all_data["nome_curso"] = all_data["nome_curso"].astype(str).str.upper()
all_data["nome_ies"] = all_data["nome_ies"].astype(str).str.upper()

# Converte colunas numéricas principais, se existirem
for col in ["concluintes", "ingressantes", "inscritos_totais", "vagas_totais"]:
    if col in all_data.columns:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce").astype("Int64")

# Remapeia 'cat_adm' e 'modalidade_ensino' para rótulos legíveis
all_data["cat_adm"] = (
    all_data["cat_adm"].replace({"1": "Federal", "2": "Estadual"}).fillna("Outro")
)
all_data["modalidade_ensino"] = (
    all_data["modalidade_ensino"]
    .replace({"1": "Presencial", "2": "EAD"})
    .fillna("Outro")
)

# =============================================================================
# Agrupamento e pivot dos dados para calcular taxas de conclusão e evasão
# =============================================================================
ingress = (
    all_data[
        [
            "nome_curso",
            "modalidade_ensino",
            "ingressantes",
            "ano",
            "concluintes",
        ]
    ]
    .groupby(["nome_curso", "modalidade_ensino", "ano"], as_index=False)
    .agg({"ingressantes": "sum", "concluintes": "sum"})
)

# Pivotar os dados para 'ingressantes' e 'concluintes'
pivot_ing = ingress.pivot_table(
    index=["nome_curso", "modalidade_ensino"],
    columns="ano",
    values="ingressantes",
)
pivot_con = ingress.pivot_table(
    index=["nome_curso", "modalidade_ensino"],
    columns="ano",
    values="concluintes",
)

# Renomeia as colunas com o padrão "ingressantes_{ano}" e "concluintes_{ano}"
pivot_ing.columns = [f"ingressantes_{ano}" for ano in pivot_ing.columns]
pivot_con.columns = [f"concluintes_{ano}" for ano in pivot_con.columns]

ingress_wide = pivot_ing.join(pivot_con).reset_index()

# =============================================================================
# Cálculo das taxas de conclusão e evasão por curso
# =============================================================================
# Fórmulas:
#   Taxa de Conclusão = concluintes_{ano} / ingressantes_{ano - D}
#   Taxa de Evasão = 1 - Taxa de Conclusão

duracoes = {
    "ENGENHARIA CIVIL": 5,
    "MEDICINA": 6,
    "DIREITO": 5,
    "ADMINISTRAÇÃO": 4,
    "Ciência da Computação": 4,
}


def calcular_metricas(df: pd.DataFrame, curso: str, duracao: int) -> pd.DataFrame:
    df_course = df[df["nome_curso"] == curso].copy()
    for ano in range(2009 + duracao, 2025):
        col_ing = f"ingressantes_{ano - duracao}"
        col_con = f"concluintes_{ano}"
        col_conclusao = f"taxa_conclusao_{ano}"
        col_evasao = f"taxa_evasao_{ano}"
        if col_ing in df_course.columns and col_con in df_course.columns:
            df_course[col_conclusao] = df_course[col_con] / df_course[col_ing]
            df_course[col_evasao] = 1 - df_course[col_conclusao]
        else:
            df_course[col_conclusao] = np.nan
            df_course[col_evasao] = np.nan
    return df_course


df_eng = calcular_metricas(
    ingress_wide, "ENGENHARIA CIVIL", duracoes["ENGENHARIA CIVIL"]
)
df_dir = calcular_metricas(ingress_wide, "DIREITO", duracoes["DIREITO"])
df_med = calcular_metricas(ingress_wide, "MEDICINA", duracoes["MEDICINA"])
df_adm = calcular_metricas(ingress_wide, "ADMINISTRAÇÃO", duracoes["ADMINISTRAÇÃO"])
df_cc = calcular_metricas(
    ingress_wide, "CIÊNCIA DA COMPUTAÇÃO", duracoes["Ciência da Computação"]
)


# Função para converter data frame wide em long para um dado prefixo e rótulo
def pivot_long_metric(df: pd.DataFrame, prefix: str, label: str) -> pd.DataFrame:
    metric_cols = [col for col in df.columns if col.startswith(prefix)]
    df_long = df.melt(
        id_vars=["nome_curso", "modalidade_ensino"],
        value_vars=metric_cols,
        var_name="ano",
        value_name="taxa",
    )
    df_long["ano"] = df_long["ano"].astype(str).str.extract(r"(\d{4})")
    df_long["tipo"] = label
    return df_long


# Criando data frames long para cada curso e para cada métrica
eng_long_conclusao = pivot_long_metric(df_eng, "taxa_conclusao_", "Taxa de Conclusão")
eng_long_evasao = pivot_long_metric(df_eng, "taxa_evasao_", "Taxa de Evasão")
eng_civil_long = pd.concat([eng_long_conclusao, eng_long_evasao])

dir_long_conclusao = pivot_long_metric(df_dir, "taxa_conclusao_", "Taxa de Conclusão")
dir_long_evasao = pivot_long_metric(df_dir, "taxa_evasao_", "Taxa de Evasão")
direito_long = pd.concat([dir_long_conclusao, dir_long_evasao])

med_long_conclusao = pivot_long_metric(df_med, "taxa_conclusao_", "Taxa de Conclusão")
med_long_evasao = pivot_long_metric(df_med, "taxa_evasao_", "Taxa de Evasão")
medicina_long = pd.concat([med_long_conclusao, med_long_evasao])

adm_long_conclusao = pivot_long_metric(df_adm, "taxa_conclusao_", "Taxa de Conclusão")
adm_long_evasao = pivot_long_metric(df_adm, "taxa_evasao_", "Taxa de Evasão")
administracao_long = pd.concat([adm_long_conclusao, adm_long_evasao])

cc_long_conclusao = pivot_long_metric(df_cc, "taxa_conclusao_", "Taxa de Conclusão")
cc_long_evasao = pivot_long_metric(df_cc, "taxa_evasao_", "Taxa de Evasão")
computacao_long = pd.concat([cc_long_conclusao, cc_long_evasao])


# =============================================================================
# Plotagem com Plotly – duas linhas: uma para "Taxa de Conclusão" (verde)
# e outra para "Taxa de Evasão" (vermelha)
# =============================================================================
def plot_taxa(df_long: pd.DataFrame, curso: str) -> None:
    fig = px.line(
        df_long,
        x="ano",
        y="taxa",
        color="tipo",
        markers=True,
        title=f"Taxa de Evasão e Conclusão ao longo dos anos - Curso de {curso}",
        labels={"ano": "Ano de Ingresso", "taxa": "Taxa"},
        color_discrete_map=color_map,
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend=dict(orientation="v", title=dict(text="Métrica")))
    fig.show()


plot_taxa(eng_civil_long, "Engenharia Civil")
plot_taxa(direito_long, "Direito")
plot_taxa(medicina_long, "Medicina")
plot_taxa(administracao_long, "Administração")
plot_taxa(computacao_long, "Ciência da Computação")
# =============================================================================
# Exportação dos dados processados para uso em outros scripts / notebooks
# =============================================================================
ingress_wide.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_ingressantes.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)
df_eng.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_eng_civil.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)
df_dir.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_direito.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)
df_med.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_medicina.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)
df_adm.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_administracao.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)
df_cc.to_csv(
    os.path.join(PASTA_PROCESSADO, "final_computacao.csv"),
    index=False,
    sep=";",
    encoding="utf-8",
)

# =============================================================================
# Auditoria após filtros básicos
# =============================================================================
df_filtrado = df_eda.copy()
auditar_df(df_filtrado, etapa="analises", contexto="apos_filtros_basicos", n=5)
