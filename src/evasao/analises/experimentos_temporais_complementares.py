from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[3]
DADOS_TEMPORAL = ROOT / "dados" / "processado" / "dados_modelagem_temporal_sbie.csv"
DIR_RESULTADOS = ROOT / "modelos" / "resultados_modelos"

CAMINHO_ABLACAO_CSV = DIR_RESULTADOS / "ablacao_sem_concluintes_t.csv"
CAMINHO_ABLACAO_MD = DIR_RESULTADOS / "ablacao_sem_concluintes_t.md"
CAMINHO_EXPANDING_CSV = DIR_RESULTADOS / "validacao_rolling_expanding_temporal.csv"
CAMINHO_EXPANDING_MD = DIR_RESULTADOS / "validacao_rolling_expanding_temporal.md"

TARGET = "taxa_evasao_alvo"
ANO_BASE = "ano_base"
ANO_ALVO = "ano_alvo"
ANO_LIMITE_TREINO = 2018
THRESHOLD_EVASAO_ALTA = 0.5

FEATURES_COMPLETAS = [
    "ingressantes_t",
    "concluintes_t",
    "matriculados_t",
    "vagas_totais_t",
    "inscritos_totais_t",
]
FEATURES_SEM_CONCLUINTES = [
    "ingressantes_t",
    "matriculados_t",
    "vagas_totais_t",
    "inscritos_totais_t",
]
COLUNAS_BASE = FEATURES_COMPLETAS + [TARGET, ANO_BASE, ANO_ALVO]
COLUNAS_PROIBIDAS_FEATURES = {
    TARGET,
    "taxa_conclusao_alvo",
    "taxa_evasao",
    "taxa_conclusao",
    "ingressantes_alvo",
    "concluintes_alvo",
    "matriculados_alvo",
}


def _formatar_valor(valor: object) -> str:
    if isinstance(valor, float):
        return f"{valor:.6f}"
    return str(valor)


def _tabela_markdown(df: pd.DataFrame) -> str:
    colunas = list(df.columns)
    linhas = [
        "| " + " | ".join(colunas) + " |",
        "| " + " | ".join("---" for _ in colunas) + " |",
    ]
    for _, row in df.iterrows():
        linhas.append(
            "| " + " | ".join(_formatar_valor(row[col]) for col in colunas) + " |"
        )
    return "\n".join(linhas)


def _salvar_markdown(
    caminho: Path,
    titulo: str,
    df: pd.DataFrame,
    nota: str,
    resumo: pd.DataFrame | None = None,
) -> None:
    partes = [f"# {titulo}", "", nota, "", _tabela_markdown(df)]
    if resumo is not None:
        partes.extend(["", "## Resumo", "", _tabela_markdown(resumo)])
    caminho.write_text("\n".join(partes) + "\n", encoding="utf-8")


def _metricas(
    y_true: pd.Series,
    y_pred: np.ndarray,
    threshold: float = THRESHOLD_EVASAO_ALTA,
) -> dict[str, float | int]:
    y_pred_clip = np.clip(y_pred, 0.0, 1.0)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred_clip >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    mse = mean_squared_error(y_true, y_pred_clip)

    return {
        "r2": r2_score(y_true, y_pred_clip),
        "mae": mean_absolute_error(y_true, y_pred_clip),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "cm_tn": int(tn),
        "cm_fp": int(fp),
        "cm_fn": int(fn),
        "cm_tp": int(tp),
    }


def carregar_base_temporal() -> pd.DataFrame:
    if not DADOS_TEMPORAL.exists():
        raise FileNotFoundError(f"Dataset temporal ausente: {DADOS_TEMPORAL}")

    df = pd.read_csv(DADOS_TEMPORAL, sep=";", usecols=COLUNAS_BASE)
    faltantes = set(COLUNAS_BASE) - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(faltantes)}")

    for features in [FEATURES_COMPLETAS, FEATURES_SEM_CONCLUINTES]:
        proibidas = set(features).intersection(COLUNAS_PROIBIDAS_FEATURES)
        if proibidas:
            raise ValueError(f"Features proibidas no experimento: {sorted(proibidas)}")

    return df.dropna(subset=COLUNAS_BASE).copy()


def executar_ablacao_sem_concluintes(df: pd.DataFrame) -> pd.DataFrame:
    treino = df[df[ANO_BASE] <= ANO_LIMITE_TREINO].copy()
    teste = df[df[ANO_BASE] > ANO_LIMITE_TREINO].copy()
    if treino.empty or teste.empty:
        raise ValueError("Split temporal vazio para a ablação sem concluintes_t.")

    especificacoes = [
        ("Random Forest completo", FEATURES_COMPLETAS),
        ("Random Forest sem concluintes_t", FEATURES_SEM_CONCLUINTES),
    ]

    linhas: list[dict[str, object]] = []
    for nome_modelo, features in especificacoes:
        modelo = RandomForestRegressor(
            n_estimators=60,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        modelo.fit(treino[features], treino[TARGET])
        pred = modelo.predict(teste[features])

        linha: dict[str, object] = {
            "experimento": "ablacao_sem_concluintes_t",
            "modelo": nome_modelo,
            "features": ",".join(features),
            "usa_concluintes_t": "concluintes_t" in features,
            "ano_limite_treino": ANO_LIMITE_TREINO,
            "linhas_treino": len(treino),
            "linhas_teste": len(teste),
            "threshold_binario": THRESHOLD_EVASAO_ALTA,
        }
        linha.update(_metricas(teste[TARGET], pred))
        linhas.append(linha)

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(CAMINHO_ABLACAO_CSV, index=False, encoding="utf-8")
    _salvar_markdown(
        CAMINHO_ABLACAO_MD,
        "Ablacao sem concluintes_t",
        resultado,
        (
            "Experimento complementar: compara o Random Forest temporal com todas "
            "as features permitidas contra uma versao sem `concluintes_t`. O "
            "pipeline principal nao foi alterado."
        ),
    )
    return resultado


def executar_validacao_expanding(df: pd.DataFrame) -> pd.DataFrame:
    anos = sorted(int(a) for a in df[ANO_BASE].dropna().unique())
    linhas: list[dict[str, object]] = []

    for ano_teste in anos:
        treino = df[df[ANO_BASE] < ano_teste].copy()
        teste = df[df[ANO_BASE] == ano_teste].copy()

        if treino[ANO_BASE].nunique() < 5 or teste.empty:
            continue

        modelos = [
            ("Regressao Linear expanding", LinearRegression()),
            (
                "Random Forest leve expanding",
                RandomForestRegressor(
                    n_estimators=30,
                    max_depth=8,
                    min_samples_split=4,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]

        for nome_modelo, modelo in modelos:
            modelo.fit(treino[FEATURES_COMPLETAS], treino[TARGET])
            pred = modelo.predict(teste[FEATURES_COMPLETAS])

            linha: dict[str, object] = {
                "experimento": "validacao_rolling_expanding_temporal",
                "modelo": nome_modelo,
                "features": ",".join(FEATURES_COMPLETAS),
                "ano_treino_min": int(treino[ANO_BASE].min()),
                "ano_treino_max": int(treino[ANO_BASE].max()),
                "ano_base_teste": int(ano_teste),
                "ano_alvo_teste_min": int(teste[ANO_ALVO].min()),
                "ano_alvo_teste_max": int(teste[ANO_ALVO].max()),
                "linhas_treino": len(treino),
                "linhas_teste": len(teste),
                "threshold_binario": THRESHOLD_EVASAO_ALTA,
            }
            linha.update(_metricas(teste[TARGET], pred))
            linhas.append(linha)

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(CAMINHO_EXPANDING_CSV, index=False, encoding="utf-8")

    resumo = (
        resultado.groupby("modelo", as_index=False)
        .agg(
            n_folds=("ano_base_teste", "count"),
            r2_medio=("r2", "mean"),
            mae_medio=("mae", "mean"),
            rmse_medio=("rmse", "mean"),
            f1_medio=("f1", "mean"),
        )
        .sort_values("modelo")
    )
    _salvar_markdown(
        CAMINHO_EXPANDING_MD,
        "Validacao rolling/expanding temporal",
        resultado,
        (
            "Experimento complementar com janela expansiva: para cada ano-base de "
            "teste, o treinamento usa apenas anos-base anteriores. O objetivo e "
            "avaliar estabilidade temporal sem alterar o pipeline principal."
        ),
        resumo=resumo,
    )
    return resultado


def main() -> None:
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = carregar_base_temporal()

    ablacao = executar_ablacao_sem_concluintes(df)
    expanding = executar_validacao_expanding(df)

    print(f"Ablacao salva em: {CAMINHO_ABLACAO_CSV}")
    print(ablacao.to_string(index=False))
    print(f"Validacao rolling/expanding salva em: {CAMINHO_EXPANDING_CSV}")
    print(
        expanding.groupby("modelo", as_index=False)
        .agg(r2_medio=("r2", "mean"), mae_medio=("mae", "mean"), f1_medio=("f1", "mean"))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
