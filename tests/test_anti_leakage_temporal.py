from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_TEMPORAL = ROOT / "dados" / "processado" / "dados_modelagem_temporal_sbie.csv"
TRATAR_DADOS = ROOT / "src" / "evasao" / "processamento_dados" / "tratar_dados.py"
RANDOMFOREST = ROOT / "src" / "evasao" / "modelagem" / "randomforest.py"
APP_EVASAO = ROOT / "src" / "evasao" / "dashboard" / "app_evasao.py"
RESULTADOS_MODELOS = ROOT / "modelos" / "resultados_modelos"
README = ROOT / "README.md"

FEATURE_COLS = {
    "ingressantes_t",
    "concluintes_t",
    "matriculados_t",
    "vagas_totais_t",
    "inscritos_totais_t",
}
FORBIDDEN_TARGET_DERIVED_COLS = {
    "taxa_evasao_alvo",
    "taxa_conclusao_alvo",
    "taxa_evasao",
    "taxa_conclusao",
    "ingressantes_alvo",
    "concluintes_alvo",
    "matriculados_alvo",
}
ANO_LIMITE_TREINO = 2018
ARTEFATOS_AUDITORIA = {
    "comparacao": RESULTADOS_MODELOS / "comparacao_baselines_temporal.csv",
    "feature_importance": RESULTADOS_MODELOS / "feature_importance_random_forest.csv",
    "metodologia": RESULTADOS_MODELOS / "metodologia_temporal_antileakage.md",
    "completude": RESULTADOS_MODELOS / "diagnostico_completude_temporal.csv",
    "balanceamento": RESULTADOS_MODELOS / "diagnostico_balanceamento_classes.csv",
}
DOC_METODOLOGIA = ROOT / "docs" / "metodologia_modelagem_temporal.md"


def _literal_string_lists(path: Path) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lists: list[set[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            values = {
                item.value
                for item in node.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if values:
                lists.append(values)
    return lists


def _assigned_string_lists(path: Path, variable_names: set[str]) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if not target_names.intersection(variable_names):
            continue
        if isinstance(node.value, ast.List):
            assigned = {
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if assigned:
                values.append(assigned)
    return values


def test_dataset_temporal_nao_expoe_colunas_derivadas_do_alvo() -> None:
    columns = set(pd.read_csv(DATASET_TEMPORAL, sep=";", nrows=0).columns)

    assert "taxa_evasao_alvo" in columns
    assert FEATURE_COLS.issubset(columns)
    assert columns.isdisjoint(FORBIDDEN_TARGET_DERIVED_COLS - {"taxa_evasao_alvo"})


def test_features_de_modelagem_nao_incluem_colunas_proibidas() -> None:
    for path in [RANDOMFOREST, APP_EVASAO]:
        for values in _literal_string_lists(path):
            if "taxa_evasao_alvo" in values or values.intersection(FEATURE_COLS):
                if values.intersection(FEATURE_COLS):
                    assert values.isdisjoint(FORBIDDEN_TARGET_DERIVED_COLS)


def test_listas_de_features_principais_sao_apenas_variaveis_em_t() -> None:
    for path in [RANDOMFOREST, APP_EVASAO]:
        listas = _assigned_string_lists(path, {"FEATURE_COLS", "feature_cols"})
        assert listas, f"Nenhuma lista de features encontrada em {path}"
        assert any(values == FEATURE_COLS for values in listas)
        for values in listas:
            if values.intersection(FEATURE_COLS):
                assert values.issubset(FEATURE_COLS)


def test_randomforest_nao_usa_train_test_split_na_modelagem_temporal() -> None:
    source = RANDOMFOREST.read_text(encoding="utf-8")

    assert "train_test_split" not in source


def test_gerador_temporal_nao_persiste_taxa_conclusao_alvo() -> None:
    source = TRATAR_DADOS.read_text(encoding="utf-8")

    assert 'df_ano["taxa_evasao_alvo"]' in source
    assert 'df_ano["taxa_conclusao_alvo"]' not in source


def test_split_temporal_tem_treino_e_teste_sem_sobreposicao() -> None:
    df = pd.read_csv(
        DATASET_TEMPORAL,
        sep=";",
        usecols=[*FEATURE_COLS, "taxa_evasao_alvo", "ano_base", "ano_alvo"],
    ).dropna()

    treino = df[df["ano_base"] <= ANO_LIMITE_TREINO]
    teste = df[df["ano_base"] > ANO_LIMITE_TREINO]

    assert not treino.empty
    assert not teste.empty
    assert treino["ano_base"].max() <= ANO_LIMITE_TREINO
    assert teste["ano_base"].min() > ANO_LIMITE_TREINO
    assert treino["ano_alvo"].max() <= teste["ano_alvo"].min()


def test_alvo_futuro_nao_reconstroi_formula_do_mesmo_ano() -> None:
    df = pd.read_csv(
        DATASET_TEMPORAL,
        sep=";",
        usecols=["ingressantes_t", "concluintes_t", "taxa_evasao_alvo"],
    ).dropna()
    df = df[df["ingressantes_t"] > 0]

    evasao_mesmo_ano = 1 - (df["concluintes_t"] / df["ingressantes_t"])
    taxa_igual = np.isclose(
        evasao_mesmo_ano,
        df["taxa_evasao_alvo"],
        rtol=0,
        atol=1e-12,
    ).mean()

    assert taxa_igual < 0.95


def test_randomforest_declara_relatorios_de_auditoria() -> None:
    source = RANDOMFOREST.read_text(encoding="utf-8")

    for caminho in ARTEFATOS_AUDITORIA.values():
        assert caminho.name in source


def test_relatorios_de_auditoria_validos_quando_modelagem_foi_executada() -> None:
    if not any(caminho.exists() for caminho in ARTEFATOS_AUDITORIA.values()):
        return

    assert all(caminho.exists() for caminho in ARTEFATOS_AUDITORIA.values())

    metricas = pd.read_csv(ARTEFATOS_AUDITORIA["comparacao"])
    assert {
        "modelo",
        "r2",
        "mae",
        "rmse",
        "mse",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cm_tn",
        "cm_fp",
        "cm_fn",
        "cm_tp",
        "linhas_teste",
    }.issubset(metricas.columns)
    assert {
        "Media historica do treino",
        "Persistencia evasao_t",
        "Regressao Linear",
        "Random Forest",
    }.issubset(set(metricas["modelo"]))
    assert metricas["linhas_teste"].nunique() == 1

    importance = pd.read_csv(ARTEFATOS_AUDITORIA["feature_importance"])
    assert {"rank", "feature", "importance"}.issubset(importance.columns)
    assert set(importance["feature"]) == FEATURE_COLS

    completude = pd.read_csv(ARTEFATOS_AUDITORIA["completude"])
    assert {
        "ano_base",
        "linhas_antes_dropna",
        "linhas_apos_dropna",
        "percentual_perda",
    }.issubset(completude.columns)

    balanceamento = pd.read_csv(ARTEFATOS_AUDITORIA["balanceamento"])
    assert {
        "conjunto",
        "prop_baixa_evasao",
        "prop_alta_evasao",
        "baseline_majoritario_accuracy",
    }.issubset(balanceamento.columns)


def test_ano_base_2020_e_diagnosticado_quando_modelagem_foi_executada() -> None:
    caminho = ARTEFATOS_AUDITORIA["completude"]
    if not caminho.exists():
        return

    diagnostico = pd.read_csv(caminho)
    linha_2020 = diagnostico[diagnostico["ano_base"] == 2020]
    assert not linha_2020.empty
    assert int(linha_2020.iloc[0]["linhas_antes_dropna"]) >= 0
    assert "percentual_perda" in linha_2020.columns


def test_docs_nao_recomendam_formula_deterministica_como_modelo_preditivo() -> None:
    textos = [
        README.read_text(encoding="utf-8").lower(),
        (ROOT / "LEIAME.md").read_text(encoding="utf-8").lower(),
        DOC_METODOLOGIA.read_text(encoding="utf-8").lower(),
    ]

    for texto in textos:
        assert "riscos metodológicos mitigados" in texto or "riscos metodologicos mitigados" in texto
        assert "taxa_conclusao -> taxa_evasao" not in texto
        assert "taxa_conclusao prediz taxa_evasao" not in texto
