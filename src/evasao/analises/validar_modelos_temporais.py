# Script para validação temporal simples de modelo preditivo.
# Agora usa a base temporal do SBIE, sem shuffle e sem leakage contemporâneo.
# Realiza:
# - auditoria do dataset temporal
# - análise de correlação numérica
# - split temporal por ano_base
# - treino de LinearRegression como baseline temporal
# - avaliação via MSE e R²
# - gráfico de comparação entre valores reais e previstos

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / "src"

# Garante que a raiz do projeto e a pasta src estejam no sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evasao.utils.inspecao import auditar_df, registrar_ambiente
from evasao.utils.paths import DADOS_PROCESSADO, MODELOS

registrar_ambiente(etapa="validacao_temporal", contexto="inicio")

CAMINHO_DADOS = DADOS_PROCESSADO / "dados_modelagem_temporal_sbie.csv"
DIR_RESULTADOS = MODELOS / "resultados_modelos"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Carrega os dados temporais do SBIE
df = pd.read_csv(CAMINHO_DADOS, sep=";", encoding="utf-8", low_memory=False)

auditar_df(df, etapa="validacao_temporal", contexto="df_temporal_bruto", n=5)

feature_cols = [
    "ingressantes_t",
    "concluintes_t",
    "matriculados_t",
    "vagas_totais_t",
    "inscritos_totais_t",
]
target_col = "taxa_evasao_alvo"

colunas_necessarias = feature_cols + [target_col, "ano_base", "ano_alvo"]
df_modelo = df[colunas_necessarias].copy()

print("\nDistribuição por ano_base antes do dropna:")
dist_pre = (
    df_modelo.groupby("ano_base", dropna=False)
    .size()
    .reset_index(name="qtd_linhas")
    .sort_values("ano_base")
)
for _, row in dist_pre.iterrows():
    print(
        f"  ano_base={int(row['ano_base'])}: {int(row['qtd_linhas']):,} linhas".replace(
            ",", "."
        )
    )

# Remove linhas sem histórico temporal completo ou sem alvo
df_modelo = df_modelo.dropna(subset=colunas_necessarias)

print("\nDistribuição por ano_base após o dropna:")
dist_pos = (
    df_modelo.groupby("ano_base", dropna=False)
    .size()
    .reset_index(name="qtd_linhas")
    .sort_values("ano_base")
)
for _, row in dist_pos.iterrows():
    print(
        f"  ano_base={int(row['ano_base'])}: {int(row['qtd_linhas']):,} linhas".replace(
            ",", "."
        )
    )

anos_pos_dropna = sorted(df_modelo["ano_base"].dropna().astype(int).unique().tolist())
if 2020 not in anos_pos_dropna:
    print("\n[ALERTA] ano_base=2020 não aparece após o dropna.")
    print(
        "[ALERTA] Isso ocorre porque as linhas de 2020 ficaram sem histórico temporal completo nas features em t."
    )

auditar_df(
    df_modelo,
    etapa="validacao_temporal",
    contexto="df_temporal_sem_nans",
    n=5,
)

print("\nCorrelação das features com o target:")
print(df_modelo[feature_cols + [target_col]].corr(numeric_only=True))

# Split temporal coerente com o artigo SBIE
ANO_CORTE = 2018
df_treino = df_modelo[df_modelo["ano_base"] <= ANO_CORTE].copy()
df_teste = df_modelo[df_modelo["ano_base"] > ANO_CORTE].copy()

if df_treino.empty or df_teste.empty:
    raise ValueError(
        "Split temporal inválido: conjunto de treino ou teste ficou vazio."
    )

auditar_df(
    df_treino,
    etapa="validacao_temporal",
    contexto="treino_temporal_ate_2018",
    n=5,
)
auditar_df(
    df_teste,
    etapa="validacao_temporal",
    contexto="teste_temporal_pos_2018",
    n=5,
)

print("\nAnos-base no treino:", sorted(df_treino["ano_base"].unique()))
print("Anos-base no teste:", sorted(df_teste["ano_base"].unique()))
print("Anos-alvo no teste:", sorted(df_teste["ano_alvo"].unique()))

# Sem shuffle: separação temporal real
X_treino = df_treino[feature_cols]
y_treino = df_treino[target_col]

X_teste = df_teste[feature_cols]
y_teste = df_teste[target_col]

# Treina baseline linear temporal
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Avalia
y_pred = modelo.predict(X_teste)

print("\nAvaliação no conjunto de TESTE temporal:")
print("Modelo avaliado: baseline linear temporal")
print(
    "Comparação principal do artigo: baseline linear temporal vs random forest temporal"
)
print("MSE:", round(mean_squared_error(y_teste, y_pred), 4))
print("R²:", round(r2_score(y_teste, y_pred), 4))

# Gráfico de comparação entre valores reais e previstos
# Para não ficar visualmente gigantesco, plota apenas uma amostra inicial.
# Observação: o eixo x representa amostras do teste temporal, não uma série única no tempo.
limite_plot = min(300, len(y_teste))

plt.figure(figsize=(12, 6))
plt.plot(
    range(limite_plot),
    y_teste.iloc[:limite_plot].values,
    label="Real",
    marker="o",
    linewidth=1,
)
plt.plot(
    range(limite_plot),
    y_pred[:limite_plot],
    label="Previsto",
    marker="x",
    linewidth=1,
)
plt.title("Comparação entre valores reais e previstos (amostra do teste temporal)")
plt.xlabel("Amostras do teste")
plt.ylabel("Taxa de Evasão Alvo")
plt.legend()
plt.grid(True)
plt.tight_layout()

caminho_figura = DIR_RESULTADOS / "validacao_temporal_comparacao.png"
plt.savefig(caminho_figura, dpi=200)
plt.close()

# Scatter: real vs previsto
plt.figure(figsize=(8, 6))
plt.scatter(y_teste, y_pred, alpha=0.35)
plt.title("Dispersão entre valores reais e previstos (teste temporal)")
plt.xlabel("Valor real")
plt.ylabel("Valor previsto")
plt.grid(True)
plt.tight_layout()

caminho_scatter = DIR_RESULTADOS / "validacao_temporal_scatter_real_vs_previsto.png"
plt.savefig(caminho_scatter, dpi=200)
plt.close()

# Histograma de resíduos
residuos = y_teste - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=40)
plt.title("Histograma de resíduos (teste temporal)")
plt.xlabel("Resíduo (real - previsto)")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()

caminho_hist = DIR_RESULTADOS / "validacao_temporal_hist_residuos.png"
plt.savefig(caminho_hist, dpi=200)
plt.close()

# Boxplot do erro por ano_base
boxplot_df = df_teste[["ano_base"]].copy()
boxplot_df["erro"] = residuos.values
plt.figure(figsize=(10, 6))
sns.boxplot(data=boxplot_df, x="ano_base", y="erro")
plt.title("Boxplot do erro por ano_base (teste temporal)")
plt.xlabel("Ano-base")
plt.ylabel("Erro (real - previsto)")
plt.grid(True)
plt.tight_layout()

caminho_boxplot = DIR_RESULTADOS / "validacao_temporal_boxplot_erro_por_ano_base.png"
plt.savefig(caminho_boxplot, dpi=200)
plt.close()

print(f"\nGráfico salvo em: {caminho_figura}")
print(f"Scatter salvo em: {caminho_scatter}")
print(f"Histograma de resíduos salvo em: {caminho_hist}")
print(f"Boxplot de erro por ano_base salvo em: {caminho_boxplot}")
