# src/evasao/rodar_pipeline.py
# Objetivo: rodar o pipeline completo sem acionar o dashboard interativo.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def run_module(module_name: str) -> None:
    print(f"\n>>> Rodando módulo: {module_name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SRC)
        if not env.get("PYTHONPATH")
        else f"{SRC}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
        cwd=ROOT,
        env=env,
    )


ETAPAS_PIPELINE = [
    # Descobre links públicos dos microdados do INEP.
    "evasao.coleta_dados.coletar_links_inep",
    # Baixa e organiza microdados brutos quando necessário.
    "evasao.coleta_dados.coleta_dados_oficiais",
    # Padroniza colunas e cria bases processadas por ano.
    "evasao.processamento_dados.pre_processamento",
    # Consolida série temporal e gera dataset temporal sem leakage.
    "evasao.processamento_dados.tratar_dados",
    # Produz análises exploratórias descritivas.
    "evasao.analises.analises",
    # Treina baselines, Regressão Linear e Random Forest no holdout temporal.
    "evasao.modelagem.randomforest",
    # Valida o baseline linear temporal e gráficos de erro.
    "evasao.analises.validar_modelos_temporais",
    # Gera gráficos finais de visualização.
    "evasao.visualizacao.gerar_graficos",
]


if __name__ == "__main__":
    for etapa in ETAPAS_PIPELINE:
        run_module(etapa)

    # Dashboard permanece manual para preservar a experiência Streamlit.
    # streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py
