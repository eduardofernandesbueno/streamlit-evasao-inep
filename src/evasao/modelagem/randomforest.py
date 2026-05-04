# # /src/evasao/modelagem/treinamento_modelo.py
# # Este script treina modelos de aprendizado de máquina para prever a taxa de evasão,
# # bem como explorar as relações entre taxa de ingresso, taxa de evasão e taxa de conclusão,
# # e realizar previsões para ingressantes e concluintes.


# -----------------------------------------------------------------------------
# 5º MODELO ADAPTADO (treina RF em tempo de execução no Streamlit)
# -----------------------------------------------------------------------------
# DISCIPLINA 2COP507 - RECONHECIMENTO DE PADRÕES
# Professores Bruno B. Zarpelão e Sylvio Barbon Jr.
# Projeto Prático de Aprendizado de Máquina com Streamlit
# Este script treina modelos temporais para prever taxa_evasao_alvo em t+1
# usando apenas variáveis disponíveis no ano-base t. Ele compara média histórica,
# persistência, Regressão Linear e Random Forest no mesmo holdout temporal.
#
# >>> IMPORTANTE PARA O DEPLOY / STREAMLIT <<<
# - NÃO commitar os arquivos .pkl no GitHub.
# - No app_evasao.py, use:
#       from evasao.modelagem.randomforest import carregar_ou_treinar_modelos_streamlit
#       modelos = carregar_ou_treinar_modelos_streamlit()
#   Isso fará com que:
#   * Se os .pkl existirem, sejam apenas carregados;
#   * Se não existirem (deploy limpo), os modelos sejam treinados em tempo de execução
#     e os .pkl salvos em modelos/modelos_salvos/ (sem ir para o Git).

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Caminhos / estrutura de pastas (baseado em evasao.utils.paths)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / "src"

# Garante que a raiz do projeto e a pasta src estejam no sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evasao.utils.inspecao import registrar_ambiente, auditar_df, auditar_csv  # noqa: E402
from evasao.utils.paths import DADOS_PROCESSADO, MODELOS  # noqa: E402

CAMINHO_DADOS_SBIE = DADOS_PROCESSADO / "dados_modelagem_temporal_sbie.csv"
CAMINHO_DADOS_BASELINE = DADOS_PROCESSADO / "dados_ingresso_evasao_conclusao.csv"

DIR_MODELOS = MODELOS / "modelos_salvos"
DIR_MODELOS.mkdir(parents=True, exist_ok=True)

DIR_RESULTADOS = MODELOS / "resultados_modelos"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Caminhos individuais dos modelos principais (.pkl)
CAMINHO_MODELO_INGRESSANTES = DIR_MODELOS / "modelo_ingressantes.pkl"
CAMINHO_MODELO_CONCLUINTES = DIR_MODELOS / "modelo_concluintes.pkl"
CAMINHO_MODELO_MELHOR = DIR_MODELOS / "modelo_melhor_evasao.pkl"
CAMINHO_MODELO_OFERTA = DIR_MODELOS / "modelo_evasao_oferta_demanda.pkl"
CAMINHO_MODELO_FLUXO = DIR_MODELOS / "modelo_evasao_fluxo_academico.pkl"
CAMINHO_MODELO_GERAL = DIR_MODELOS / "modelo_evasao_geral_compacto.pkl"

CAMINHO_COMPARACAO_BASELINES_CSV = DIR_RESULTADOS / "comparacao_baselines_temporal.csv"
CAMINHO_COMPARACAO_BASELINES_MD = DIR_RESULTADOS / "comparacao_baselines_temporal.md"
CAMINHO_FEATURE_IMPORTANCE_CSV = DIR_RESULTADOS / "feature_importance_random_forest.csv"
CAMINHO_FEATURE_IMPORTANCE_MD = DIR_RESULTADOS / "feature_importance_random_forest.md"
CAMINHO_METODOLOGIA_MD = DIR_RESULTADOS / "metodologia_temporal_antileakage.md"


def _salvar_markdown_tabela(df: pd.DataFrame, caminho: Path, titulo: str) -> None:
    linhas = [f"# {titulo}", ""]
    linhas.append("| " + " | ".join(df.columns) + " |")
    linhas.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        valores = []
        for valor in row:
            if isinstance(valor, float):
                valores.append(f"{valor:.6f}")
            else:
                valores.append(str(valor))
        linhas.append("| " + " | ".join(valores) + " |")

    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _calcular_metricas_modelo(
    nome_modelo: str,
    y_true: pd.Series,
    y_pred,
    threshold: float,
) -> dict[str, float | str]:
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred_arr)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (np.clip(y_pred_arr, 0.0, 1.0) >= threshold).astype(int)

    return {
        "modelo": nome_modelo,
        "r2": r2_score(y_true, y_pred_arr),
        "mae": mean_absolute_error(y_true, y_pred_arr),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }


def _predizer_persistencia_evasao_t(X_test: pd.DataFrame, fallback: float) -> np.ndarray:
    ingressantes = pd.to_numeric(X_test["ingressantes_t"], errors="coerce")
    concluintes = pd.to_numeric(X_test["concluintes_t"], errors="coerce")
    evasao_t = 1 - (concluintes / ingressantes)
    evasao_t = evasao_t.replace([np.inf, -np.inf], np.nan).fillna(fallback)
    return np.clip(evasao_t.to_numpy(dtype=float), 0.0, 1.0)


def _salvar_documentacao_metodologica() -> None:
    texto = """# Metodologia Temporal e Anti-Leakage

## Problema Original

A formulação original tinha risco de data leakage porque a taxa de evasão podia ser
reconstruída diretamente a partir de variáveis contemporâneas, especialmente
ingressantes, concluintes e taxa de conclusão.

## Formulação Atual

O dataset atual usa variáveis disponíveis no ano-base t para prever
`taxa_evasao_alvo` no ano t+1. O holdout temporal é fixo: treino com
`ano_base <= 2018` e teste com `ano_base > 2018`.

## Variáveis Proibidas

`taxa_conclusao_alvo`, `ingressantes_alvo`, `concluintes_alvo` e quaisquer
variáveis futuras ou derivadas diretamente do alvo não podem entrar em X.

## Papel dos Baselines

Os baselines tornam a comparação auditável:

- média histórica do treino;
- persistência (`taxa_evasao_t`);
- Regressão Linear;
- Random Forest.

O Random Forest só deve ser defendido cientificamente quando superar esses
baselines no mesmo holdout temporal.
"""
    CAMINHO_METODOLOGIA_MD.write_text(texto, encoding="utf-8")
    print(f"Documentação metodológica salva em: {CAMINHO_METODOLOGIA_MD}")


# -----------------------------------------------------------------------------
# 1) Função de treinamento completo (usada em CLI e também pelo Streamlit)
# -----------------------------------------------------------------------------
def treinar_modelos(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    threshold_evasao_alta: float = 0.5,
):
    """
    Treina modelos para:
    1) Prever número de ingressantes e concluintes em base auxiliar descritiva.
    2) Prever taxa de evasão futura em base temporal SBIE (Regressão Linear x Random Forest),
       selecionando automaticamente o melhor modelo.

    O script agora:
    - usa como base principal o dataset temporal SBIE sem leakage contemporâneo;
    - utiliza split temporal por ano_base para avaliar capacidade de predição futura.

    Além disso:
    - Salva os modelos em BASE_DIR/modelos/modelos_salvos
    - Salva métricas em BASE_DIR/modelos/resultados_modelos/metricas_modelos.txt
    - Gera e salva a matriz de confusão binária em BASE_DIR/modelos/resultados_modelos/matriz_confusao.png
    - Imprime um resumo dos resultados no terminal em uma seção
      "Exibição de resultados", inspirada no exemplo KNN do Prof. Zarpelão.
    """
    import time

    inicio_execucao = time.time()
    print(f"[INÍCIO] Treinamento iniciado às: {time.strftime('%H:%M:%S')}")
    registrar_ambiente(etapa="modelagem", contexto="inicio_modelagem_base")

    # Base principal do artigo SBIE: dataset temporal sem leakage
    caminho_dados = CAMINHO_DADOS_SBIE
    df = pd.read_csv(caminho_dados, sep=";", encoding="utf-8", low_memory=False)

    # Base auxiliar/descritiva: usada apenas para tarefas secundárias
    caminho_dados_baseline = CAMINHO_DADOS_BASELINE
    df_baseline = pd.read_csv(
        caminho_dados_baseline,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )

    auditar_df(df, etapa="modelagem_temporal_sbie", contexto="df_temporal_carregado", n=5)
    auditar_df(df_baseline, etapa="modelagem_baseline", contexto="df_baseline_carregado", n=5)

    if "ano_base" in df.columns:
        df_treino = df[df["ano_base"] <= 2018]
        df_teste = df[df["ano_base"] > 2018]

        auditar_df(df_treino, etapa="modelagem_temporal_sbie", contexto="treino_ate_2018", n=5)
        auditar_df(df_teste, etapa="modelagem_temporal_sbie", contexto="teste_2019_2024", n=5)

    # Definir diretório para salvar modelos
    caminho_modelo = DIR_MODELOS
    os.makedirs(caminho_modelo, exist_ok=True)

    resultados = {
        "modelos_ingressantes_concluintes": {},
        "modelo_evasao_regressao": {},
        "classificacao_evasao_binaria": {},
        "caminhos": {},
    }

    # ------------------------------------------------------------------
    # 1) Treinamento para prever ingressantes e concluintes
    # ------------------------------------------------------------------
    if all(
        col in df_baseline.columns
        for col in [
            "numero_cursos",
            "vagas_totais",
            "inscritos_totais",
            "ingressantes",
            "concluintes",
        ]
    ):
        print("Treinando modelos para ingressantes e concluintes...")

        # Features e targets para previsões de ingressantes e concluintes
        X = df_baseline[["numero_cursos", "vagas_totais", "inscritos_totais"]]
        y_ingressantes = df_baseline["ingressantes"]
        y_concluintes = df_baseline["concluintes"]

        # Divisão dos dados para ingressantes
        X_train_ing, X_test_ing, y_train_ing, y_test_ing = train_test_split(
            X, y_ingressantes, test_size=0.2, random_state=42
        )

        # Modelo Random Forest para ingressantes
        modelo_ingressantes = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        modelo_ingressantes.fit(X_train_ing, y_train_ing)
        score_ingressantes = modelo_ingressantes.score(X_test_ing, y_test_ing)
        print(f"Score R² para ingressantes (Random Forest): {score_ingressantes:.4f}")

        # Divisão dos dados para concluintes
        X_train_conc, X_test_conc, y_train_conc, y_test_conc = train_test_split(
            X, y_concluintes, test_size=0.2, random_state=42
        )

        # Modelo Random Forest para concluintes
        modelo_concluintes = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        modelo_concluintes.fit(X_train_conc, y_train_conc)
        score_concluintes = modelo_concluintes.score(X_test_conc, y_test_conc)
        print(f"Score R² para concluintes (Random Forest): {score_concluintes:.4f}")

        # Salvar modelos de ingressantes e concluintes
        caminho_modelo_ing = CAMINHO_MODELO_INGRESSANTES
        caminho_modelo_conc = CAMINHO_MODELO_CONCLUINTES
        joblib.dump(modelo_ingressantes, caminho_modelo_ing)
        joblib.dump(modelo_concluintes, caminho_modelo_conc)

        resultados["modelos_ingressantes_concluintes"] = {
            "score_r2_ingressantes": score_ingressantes,
            "score_r2_concluintes": score_concluintes,
            "caminho_modelo_ingressantes": str(caminho_modelo_ing),
            "caminho_modelo_concluintes": str(caminho_modelo_conc),
        }
    else:
        print(
            "Aviso: colunas para ingressantes/concluintes não encontradas. "
            "Pulando etapa de modelos de ingresso/conclusão."
        )

    # ------------------------------------------------------------------
    # 2) Treinamento para taxas de evasão (Regressão)
    # ------------------------------------------------------------------
    print("\nTreinando modelos para taxa de evasão (base temporal SBIE)...")

    # Features temporais em t para prever evasão no ano alvo (t+1)
    feature_cols = [
        "ingressantes_t",
        "concluintes_t",
        "matriculados_t",
        "vagas_totais_t",
        "inscritos_totais_t",
    ]

    colunas_relevantes = feature_cols + ["taxa_evasao_alvo", "ano_base", "ano_alvo"]
    df_reg = df[colunas_relevantes].copy()

    print("\nResumo pré-dropna por ano_base:")
    dist_pre = (
        df_reg.groupby("ano_base", dropna=False)
        .size()
        .reset_index(name="qtd_linhas")
        .sort_values("ano_base")
    )
    for _, row in dist_pre.iterrows():
        print(
            f"  ano_base={int(row['ano_base'])}: {int(row['qtd_linhas']):,} linhas".replace(",", ".")
        )

    # Remove linhas sem histórico temporal completo ou sem alvo
    df_reg = df_reg.dropna(
        subset=feature_cols + ["taxa_evasao_alvo", "ano_base", "ano_alvo"]
    )

    ano_limite_treino = 2018

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO APÓS dropna() DA BASE TEMPORAL SBIE")
    print("=" * 70)
    print(f"Linhas restantes em df_reg: {len(df_reg):,}".replace(",", "."))

    dist_ano_base = (
        df_reg.groupby("ano_base", dropna=False)
        .size()
        .reset_index(name="qtd_linhas")
        .sort_values("ano_base")
    )

    print("\nLinhas restantes por ano_base:")
    for _, row in dist_ano_base.iterrows():
        ano_base = int(row["ano_base"])
        qtd = f"{int(row['qtd_linhas']):,}".replace(",", ".")
        print(f"  ano_base={ano_base}: {qtd} linhas")

    anos_esperados_teste = sorted(
        [int(a) for a in df_reg["ano_base"].dropna().unique() if int(a) > ano_limite_treino]
    )
    print(f"\nAnos-base efetivamente disponíveis para teste após dropna: {anos_esperados_teste}")

    if 2020 not in anos_esperados_teste:
        print("\n[ALERTA] ano_base=2020 não restou em df_reg após dropna().")
        print(
            "[ALERTA] Isso indica que todas as linhas de 2020 foram eliminadas por falta de pelo menos uma feature temporal ou do alvo."
        )

        df_2020_original = df[df["ano_base"] == 2020].copy()
        print(
            f"Linhas originais com ano_base=2020 antes do dropna: {len(df_2020_original):,}".replace(",", ".")
        )

        if not df_2020_original.empty:
            colunas_diag = feature_cols + ["taxa_evasao_alvo", "ano_base", "ano_alvo"]
            faltantes_2020 = (
                df_2020_original[colunas_diag].isna().sum().sort_values(ascending=False)
            )

            print("\nNaNs por coluna em ano_base=2020 antes do dropna:")
            for col, qtd in faltantes_2020.items():
                print(f"  {col}: {int(qtd):,}".replace(",", "."))

            mascara_completa_2020 = df_2020_original[colunas_diag].notna().all(axis=1)
            linhas_validas_2020 = int(mascara_completa_2020.sum())
            print(
                f"\nLinhas completas de 2020 que sobreviveriam ao dropna: {linhas_validas_2020:,}".replace(",", ".")
            )

    print("=" * 70 + "\n")

    # Split temporal obrigatório: todos os modelos usam o mesmo holdout.
    if "ano_base" not in df_reg.columns:
        raise ValueError("Coluna 'ano_base' ausente: validação temporal é obrigatória.")

    df_treino = df_reg[df_reg["ano_base"] <= ano_limite_treino]
    df_teste = df_reg[df_reg["ano_base"] > ano_limite_treino]

    if df_treino.empty or df_teste.empty:
        raise ValueError(
            "Split temporal inválido: treino ou teste ficou vazio após dropna()."
        )

    print(
        f"Split temporal aplicado para evasão SBIE: treino até ano_base {ano_limite_treino}, "
        "teste em anos-base posteriores."
    )
    print(
        "Anos-base no treino:",
        sorted(df_treino["ano_base"].unique()),
        "| Anos-base no teste:",
        sorted(df_teste["ano_base"].unique()),
    )
    X_train = df_treino[feature_cols]
    y_train = df_treino["taxa_evasao_alvo"]
    X_test = df_teste[feature_cols]
    y_test = df_teste["taxa_evasao_alvo"]

    # Baseline 1: média histórica do treino
    media_historica_treino = float(y_train.mean())
    y_pred_media = np.full(len(y_test), media_historica_treino)

    # Baseline 2: persistência da evasão observada em t
    y_pred_persistencia = _predizer_persistencia_evasao_t(
        X_test,
        fallback=media_historica_treino,
    )

    # Modelo 3: Regressão Linear
    modelo_linear = LinearRegression()
    modelo_linear.fit(X_train, y_train)
    y_pred_linear = modelo_linear.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    rmse_linear = float(np.sqrt(mse_linear))
    r2_linear = r2_score(y_test, y_pred_linear)

    # Modelo 4: Random Forest
    modelo_rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = float(np.sqrt(mse_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    metricas_comparacao = pd.DataFrame(
        [
            _calcular_metricas_modelo(
                "Media historica do treino",
                y_test,
                y_pred_media,
                threshold_evasao_alta,
            ),
            _calcular_metricas_modelo(
                "Persistencia evasao_t",
                y_test,
                y_pred_persistencia,
                threshold_evasao_alta,
            ),
            _calcular_metricas_modelo(
                "Regressao Linear",
                y_test,
                y_pred_linear,
                threshold_evasao_alta,
            ),
            _calcular_metricas_modelo(
                "Random Forest",
                y_test,
                y_pred_rf,
                threshold_evasao_alta,
            ),
        ]
    )
    metricas_comparacao.insert(1, "ano_limite_treino", ano_limite_treino)
    metricas_comparacao.insert(2, "linhas_treino", len(df_treino))
    metricas_comparacao.insert(3, "linhas_teste", len(df_teste))
    metricas_comparacao.to_csv(
        CAMINHO_COMPARACAO_BASELINES_CSV,
        index=False,
        encoding="utf-8",
    )
    _salvar_markdown_tabela(
        metricas_comparacao,
        CAMINHO_COMPARACAO_BASELINES_MD,
        "Comparacao Temporal: Baselines vs Random Forest",
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": modelo_rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.insert(0, "rank", range(1, len(feature_importance) + 1))
    feature_importance.to_csv(
        CAMINHO_FEATURE_IMPORTANCE_CSV,
        index=False,
        encoding="utf-8",
    )
    _salvar_markdown_tabela(
        feature_importance,
        CAMINHO_FEATURE_IMPORTANCE_MD,
        "Importancia das Variaveis - Random Forest Temporal",
    )
    _salvar_documentacao_metodologica()

    # Escolher o melhor modelo baseado no R²
    if r2_rf > r2_linear:
        melhor_modelo = modelo_rf
        nome_melhor_modelo = "Random Forest"
    else:
        melhor_modelo = modelo_linear
        nome_melhor_modelo = "Regressão Linear"

    # Salvar o modelo escolhido
    caminho_modelo_evasao = CAMINHO_MODELO_MELHOR
    joblib.dump(melhor_modelo, caminho_modelo_evasao)

    # ------------------------------------------------------------------
    # Exibição de resultados no terminal (estilo Prof. Zarpelão)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Exibição de resultados: validação temporal para taxa de evasão")
    print("=" * 70)
    print(metricas_comparacao.to_string(index=False))
    print(f"\nMelhor modelo (critério R²): {nome_melhor_modelo}")
    print(f"Modelo salvo em: {caminho_modelo_evasao}")
    print(f"Métricas comparativas salvas em: {CAMINHO_COMPARACAO_BASELINES_CSV}")
    print(f"Métricas comparativas Markdown salvas em: {CAMINHO_COMPARACAO_BASELINES_MD}")
    print(f"Feature importance salva em: {CAMINHO_FEATURE_IMPORTANCE_CSV}")
    print(f"Feature importance Markdown salva em: {CAMINHO_FEATURE_IMPORTANCE_MD}")

    resultados["modelo_evasao_regressao"] = {
        "mae_linear": mae_linear,
        "rmse_linear": rmse_linear,
        "mse_linear": mse_linear,
        "r2_linear": r2_linear,
        "mae_random_forest": mae_rf,
        "rmse_random_forest": rmse_rf,
        "mse_random_forest": mse_rf,
        "r2_random_forest": r2_rf,
        "melhor_modelo": nome_melhor_modelo,
        "caminho_modelo": str(caminho_modelo_evasao),
        "caminho_comparacao_baselines_csv": str(CAMINHO_COMPARACAO_BASELINES_CSV),
        "caminho_comparacao_baselines_md": str(CAMINHO_COMPARACAO_BASELINES_MD),
    }
    resultados["feature_importance_random_forest"] = {
        "ranking": feature_importance.to_dict(orient="records"),
        "caminho_csv": str(CAMINHO_FEATURE_IMPORTANCE_CSV),
        "caminho_md": str(CAMINHO_FEATURE_IMPORTANCE_MD),
    }
    resultados["documentacao_metodologica"] = str(CAMINHO_METODOLOGIA_MD)

    # ------------------------------------------------------------------
    # 3) Análise binária da evasão (matriz de confusão + métricas)
    # ------------------------------------------------------------------
    caminho_metricas = DIR_RESULTADOS
    os.makedirs(caminho_metricas, exist_ok=True)

    # Geração da Matriz de Confusão (binarizando a taxa de evasão com threshold parametrizado)
    threshold = threshold_evasao_alta
    y_test_class = (y_test >= threshold).astype(int)
    y_pred_class = (melhor_modelo.predict(X_test) >= threshold).astype(int)

    cm = confusion_matrix(y_test_class, y_pred_class)
    acc = accuracy_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class, zero_division=0)
    rec = recall_score(y_test_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

    print("\n" + "=" * 70)
    print(
        f"Exibição de resultados: Classificação binária da evasão "
        f"(threshold {threshold:.2f})"
    )
    print("=" * 70)
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nMatriz de Confusão (linhas = verdadeiro, colunas = predito):")
    print(cm)

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test_class,
            y_pred_class,
            target_names=["Baixa evasão", "Alta evasão"],
            zero_division=0,
        )
    )

    # Salvar as métricas dos modelos em arquivo texto (regressão + classificação)
    caminho_metricas_txt = caminho_metricas / "metricas_modelos.txt"
    with open(caminho_metricas_txt, "w") as f:
        f.write(f"Modelo de evasão escolhido: {nome_melhor_modelo}\n\n")
        f.write("=== Comparação temporal: baselines vs modelos ===\n")
        f.write(metricas_comparacao.to_string(index=False))
        f.write("\n\n")
        f.write(f"CSV comparativo: {CAMINHO_COMPARACAO_BASELINES_CSV}\n")
        f.write(f"Markdown comparativo: {CAMINHO_COMPARACAO_BASELINES_MD}\n")
        f.write(f"Feature importance CSV: {CAMINHO_FEATURE_IMPORTANCE_CSV}\n")
        f.write(f"Feature importance Markdown: {CAMINHO_FEATURE_IMPORTANCE_MD}\n")
        f.write(f"Documentação metodológica: {CAMINHO_METODOLOGIA_MD}\n\n")

        f.write(f"=== Classificação binária da evasão (threshold {threshold:.2f}) ===\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão: {prec:.4f}\n")
        f.write(f"Recall:   {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")

        f.write("Matriz de confusão:\n")
        f.write(str(cm) + "\n\n")

        f.write("Relatório de classificação:\n")
        f.write(
            classification_report(
                y_test_class,
                y_pred_class,
                target_names=["Baixa evasão", "Alta evasão"],
                zero_division=0,
            )
        )

    # Salvar figura da matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome_melhor_modelo}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    caminho_cm = caminho_metricas / "matriz_confusao.png"
    plt.tight_layout()
    plt.savefig(caminho_cm)
    plt.close()

    print(f"\nArquivos de métricas salvos em: {caminho_metricas_txt}")
    print(f"Figura da matriz de confusão salva em: {caminho_cm}")
    resultados["classificacao_evasao_binaria"] = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "matriz_confusao": cm.tolist(),
        "caminho_metricas_txt": str(caminho_metricas_txt),
        "caminho_matriz_confusao_png": str(caminho_cm),
    }

    resultados["caminhos"]["metricas"] = str(caminho_metricas)
    resultados["caminhos"]["modelos"] = str(caminho_modelo)
    resultados["caminhos"]["comparacao_baselines_csv"] = str(
        CAMINHO_COMPARACAO_BASELINES_CSV
    )
    resultados["caminhos"]["feature_importance_csv"] = str(CAMINHO_FEATURE_IMPORTANCE_CSV)
    resultados["caminhos"]["metodologia_temporal"] = str(CAMINHO_METODOLOGIA_MD)

    # ------------------------------------------------------------------
    # 4) Treinamento de modelos de evasão por subconjuntos de variáveis
    #    (baseado nos subconjuntos definidos em pre_processamento.py)
    # ------------------------------------------------------------------
    subconjuntos_features = {
        # Oferta e demanda temporal em t
        "oferta_demanda": [
            "vagas_totais_t",
            "inscritos_totais_t",
            "ingressantes_t",
        ],
        # Fluxo acadêmico temporal em t
        "fluxo_academico": [
            "ingressantes_t",
            "matriculados_t",
            "concluintes_t",
        ],
        # Visão geral temporal compacta em t
        "geral_compacto": [
            "ingressantes_t",
            "concluintes_t",
            "matriculados_t",
            "vagas_totais_t",
            "inscritos_totais_t",
        ],
    }

    # Arquivo adicional com métricas por subconjunto
    caminho_metricas_sub = caminho_metricas / "metricas_modelos_subconjuntos.txt"
    with open(caminho_metricas_sub, "w") as f_sub:
        f_sub.write("Resultados por subconjunto de variáveis (taxa_evasao)\n")
        f_sub.write("=" * 70 + "\n\n")

    resultados["subconjuntos_evasao"] = {}
    resultados["caminhos"]["metricas_subconjuntos"] = str(caminho_metricas_sub)

    for nome_sub, cols in subconjuntos_features.items():
        # Garante que as colunas existem no dataframe temporal
        cols_validas = [c for c in cols if c in df.columns]
        if not cols_validas:
            print(
                f"Aviso: subconjunto '{nome_sub}' sem colunas válidas em df. Pulando."
            )
            continue

        print(
            f"\nTreinando modelo de evasão para subconjunto: {nome_sub} "
            f"com features: {cols_validas}"
        )

        colunas_relevantes_sub = cols_validas + ["taxa_evasao_alvo", "ano_base", "ano_alvo"]
        df_sub = df[colunas_relevantes_sub].dropna(subset=cols_validas + ["taxa_evasao_alvo", "ano_base", "ano_alvo"])

        if df_sub.empty:
            print(
                f"Aviso: subconjunto '{nome_sub}' ficou vazio após remover NaNs das features temporais. Pulando."
            )
            continue

        # Split temporal correto com ano_base
        if "ano_base" in df_sub.columns:
            ano_limite_treino_sub = 2018
            df_treino_sub = df_sub[df_sub["ano_base"] <= ano_limite_treino_sub]
            df_teste_sub = df_sub[df_sub["ano_base"] > ano_limite_treino_sub]

            if df_treino_sub.empty or df_teste_sub.empty:
                print(
                    f"Aviso: split temporal por ano_base não pôde ser aplicado para '{nome_sub}'. "
                    "Usando hold-out aleatório."
                )
                X_sub = df_sub[cols_validas]
                y_sub = df_sub["taxa_evasao_alvo"]
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                    X_sub, y_sub, test_size=0.2, random_state=42
                )
            else:
                print(
                    f"Split temporal para '{nome_sub}': treino até ano_base {ano_limite_treino_sub}, "
                    "teste em anos-base posteriores."
                )
                X_train_sub = df_treino_sub[cols_validas]
                y_train_sub = df_treino_sub["taxa_evasao_alvo"]
                X_test_sub = df_teste_sub[cols_validas]
                y_test_sub = df_teste_sub["taxa_evasao_alvo"]
        else:
            print(
                f"Aviso: coluna 'ano_base' não encontrada em df_sub ({nome_sub}). "
                "Usando hold-out aleatório."
            )
            X_sub = df_sub[cols_validas]
            y_sub = df_sub["taxa_evasao_alvo"]
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=42
            )

        # Modelo 1: Regressão Linear
        modelo_linear_sub = LinearRegression()
        modelo_linear_sub.fit(X_train_sub, y_train_sub)
        y_pred_linear_sub = modelo_linear_sub.predict(X_test_sub)
        mse_linear_sub = mean_squared_error(y_test_sub, y_pred_linear_sub)
        r2_linear_sub = r2_score(y_test_sub, y_pred_linear_sub)

        # Modelo 2: Random Forest
        modelo_rf_sub = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        modelo_rf_sub.fit(X_train_sub, y_train_sub)
        y_pred_rf_sub = modelo_rf_sub.predict(X_test_sub)
        mse_rf_sub = mean_squared_error(y_test_sub, y_pred_rf_sub)
        r2_rf_sub = r2_score(y_test_sub, y_pred_rf_sub)

        if r2_rf_sub > r2_linear_sub:
            melhor_sub = modelo_rf_sub
            nome_melhor_sub = "Random Forest"
        else:
            melhor_sub = modelo_linear_sub
            nome_melhor_sub = "Regressão Linear"

        caminho_modelo_sub = DIR_MODELOS / f"modelo_evasao_{nome_sub}.pkl"
        joblib.dump(melhor_sub, caminho_modelo_sub)

        # Classificação binária para o subconjunto (mesmo limiar da parte geral)
        y_test_class_sub = (y_test_sub >= threshold).astype(int)
        y_pred_class_sub = (melhor_sub.predict(X_test_sub) >= threshold).astype(int)
        cm_sub = confusion_matrix(y_test_class_sub, y_pred_class_sub)
        acc_sub = accuracy_score(y_test_class_sub, y_pred_class_sub)
        prec_sub = precision_score(
            y_test_class_sub, y_pred_class_sub, zero_division=0
        )
        rec_sub = recall_score(y_test_class_sub, y_pred_class_sub, zero_division=0)
        f1_sub = f1_score(y_test_class_sub, y_pred_class_sub, zero_division=0)

        # Persistência de métricas por subconjunto
        with open(caminho_metricas_sub, "a") as f_sub:
            f_sub.write(f"Subconjunto: {nome_sub}\n")
            f_sub.write(f"Features: {cols_validas}\n")
            f_sub.write(
                f"Regressão Linear  -> MSE: {mse_linear_sub:.4f}, "
                f"R²: {r2_linear_sub:.4f}\n"
            )
            f_sub.write(
                f"Random Forest     -> MSE: {mse_rf_sub:.4f}, "
                f"R²: {r2_rf_sub:.4f}\n"
            )
            f_sub.write(f"Melhor modelo: {nome_melhor_sub}\n")
            f_sub.write(
                f"Classificação binária (threshold {threshold:.2f}) -> "
                f"Acurácia: {acc_sub:.4f}, Precisão: {prec_sub:.4f}, "
                f"Recall: {rec_sub:.4f}, F1: {f1_sub:.4f}\n"
            )
            f_sub.write(f"Matriz de confusão:\n{cm_sub}\n")
            f_sub.write("-" * 70 + "\n\n")

        resultados["subconjuntos_evasao"][nome_sub] = {
            "features": cols_validas,
            "mse_linear": mse_linear_sub,
            "r2_linear": r2_linear_sub,
            "mse_random_forest": mse_rf_sub,
            "r2_random_forest": r2_rf_sub,
            "melhor_modelo": nome_melhor_sub,
            "caminho_modelo": str(caminho_modelo_sub),
            "accuracy": acc_sub,
            "precision": prec_sub,
            "recall": rec_sub,
            "f1": f1_sub,
            "matriz_confusao": cm_sub.tolist(),
        }

    print("\nTreinamento concluído.")
    fim_execucao = time.time()
    duracao = fim_execucao - inicio_execucao
    print(f"[FIM] Treinamento finalizado às: {time.strftime('%H:%M:%S')}")
    print(f"⏱️ Tempo total de execução: {duracao:.2f} segundos")
    return resultados


# -----------------------------------------------------------------------------
# 2) Camada para uso no Streamlit (treino/carregamento em tempo de execução)
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados_streamlit() -> pd.DataFrame:
    """
    Carrega o CSV temporal principal do SBIE
    com cache no Streamlit.
    """
    return pd.read_csv(
        CAMINHO_DADOS_SBIE,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )


@st.cache_resource
def carregar_ou_treinar_modelos_streamlit(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    threshold_evasao_alta: float = 0.5,
):
    """
    Usado pelo app Streamlit (app_evasao.py):

    - Se os arquivos .pkl já existirem em modelos/modelos_salvos/,
      apenas carrega todos;
    - Se não existirem (como no deploy do Streamlit Cloud),
      chama treinar_modelos(...) em tempo de execução, que:
        * lê o CSV principal em dados/processado/dados_modelagem_temporal_sbie.csv
        * treina todos os modelos (ingressantes, concluintes, evasão geral
          e subconjuntos)
        * salva os .pkl em modelos/modelos_salvos/
      e, em seguida, carrega os .pkl recém-criados.

    O resultado é um dicionário com todos os modelos relevantes.
    """

    caminhos_esperados = {
        "modelo_ingressantes": CAMINHO_MODELO_INGRESSANTES,
        "modelo_concluintes": CAMINHO_MODELO_CONCLUINTES,
        "modelo_melhor_evasao": CAMINHO_MODELO_MELHOR,
        "modelo_evasao_oferta_demanda": CAMINHO_MODELO_OFERTA,
        "modelo_evasao_fluxo_academico": CAMINHO_MODELO_FLUXO,
        "modelo_evasao_geral_compacto": CAMINHO_MODELO_GERAL,
    }

    # Se todos os modelos já existem no disco, só carrega (rápido)
    if all(path.exists() for path in caminhos_esperados.values()):
        modelos = {nome: joblib.load(path) for nome, path in caminhos_esperados.items()}
        return modelos

    # Caso contrário, treina em tempo de execução
    st.warning(
        "Modelos pré-treinados não encontrados no container. "
        "Treinando modelos de evasão em tempo real (pode levar alguns segundos)."
    )

    # IMPORTANTE: treinar_modelos usa CAMINHO_DADOS / DIR_MODELOS relativos a BASE_DIR,
    # então assumimos que o CWD do app é a raiz do projeto (BASE_DIR),
    # que é o comportamento padrão do Streamlit Cloud.
    treinar_modelos(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        threshold_evasao_alta=threshold_evasao_alta,
    )

    # Depois do treino, carrega o que foi salvo
    modelos = {}
    for nome, path in caminhos_esperados.items():
        if path.exists():
            modelos[nome] = joblib.load(path)
        else:
            st.error(f"Falha ao localizar modelo salvo: {path}")
    return modelos


# -----------------------------------------------------------------------------
# Execução em linha de comando (opcional, para testes locais)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    treinar_modelos()
