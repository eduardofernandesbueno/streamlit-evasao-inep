from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "modelos" / "resultados_modelos"


def test_artefato_feature_based_esta_marcado_como_legado() -> None:
    texto = (
        RESULTADOS / "metricas_modelos_feature_based.txt"
    ).read_text(encoding="utf-8")

    assert texto.startswith("LEGADO / NAO USAR COMO EVIDENCIA ATUAL")
    assert "comparacao_baselines_temporal" in texto


def test_prever_com_modelos_nao_sugere_taxas_contemporaneas() -> None:
    fonte = (
        ROOT / "src" / "evasao" / "analises" / "prever_com_modelos.py"
    ).read_text(encoding="utf-8").lower()

    assert "taxa_ingresso" not in fonte
    assert "taxa_conclusao" not in fonte
    assert "concluintes_t" in fonte
    assert "ingressantes_t" in fonte


def test_relatorio_pdf_nao_cita_acuracia_perfeita_atual() -> None:
    fonte = (
        ROOT / "src" / "evasao" / "relatorio" / "gerar_relatorio_pdf.py"
    ).read_text(encoding="utf-8").lower()

    assert "100% acur" not in fonte
    assert "validação temporal" in fonte or "validacao temporal" in fonte


def test_ablacao_sem_concluintes_t_tem_metricas_esperadas() -> None:
    caminho = RESULTADOS / "ablacao_sem_concluintes_t.csv"
    assert caminho.exists()

    df = pd.read_csv(caminho)
    assert {
        "modelo",
        "features",
        "usa_concluintes_t",
        "r2",
        "mae",
        "rmse",
        "mse",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }.issubset(df.columns)
    assert {"Random Forest completo", "Random Forest sem concluintes_t"}.issubset(
        set(df["modelo"])
    )

    sem_concluintes = df[df["modelo"] == "Random Forest sem concluintes_t"].iloc[0]
    assert not bool(sem_concluintes["usa_concluintes_t"])
    assert "concluintes_t" not in sem_concluintes["features"]


def test_validacao_rolling_expanding_temporal_tem_folds() -> None:
    caminho = RESULTADOS / "validacao_rolling_expanding_temporal.csv"
    assert caminho.exists()

    df = pd.read_csv(caminho)
    assert {
        "modelo",
        "ano_treino_min",
        "ano_treino_max",
        "ano_base_teste",
        "r2",
        "mae",
        "rmse",
        "mse",
        "f1",
    }.issubset(df.columns)
    assert set(df["modelo"]).issuperset(
        {"Regressao Linear expanding", "Random Forest leve expanding"}
    )
    assert (df["ano_treino_max"] < df["ano_base_teste"]).all()
    assert df["ano_base_teste"].nunique() >= 3
