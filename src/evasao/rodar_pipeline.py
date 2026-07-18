# src/evasao/rodar_pipeline.py
# Objetivo: orquestrar o pipeline principal sem acionar o dashboard interativo.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def run_module(module_name: str) -> None:
    print(f"\n>>> Rodando módulo: {module_name}")
    run_command([sys.executable, "-m", module_name], label=module_name)


def run_command(command: list[str], label: str) -> None:
    print(f"\n>>> Rodando etapa: {label}")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SRC)
        if not env.get("PYTHONPATH")
        else f"{SRC}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=env,
    )


ETAPAS_COLETA = [
    # Descobre links públicos dos microdados do INEP.
    "evasao.coleta_dados.coletar_links_inep",
    # Baixa e organiza microdados brutos quando necessário.
    "evasao.coleta_dados.coleta_dados_oficiais",
]

ETAPAS_PIPELINE_PRINCIPAL = [
    # Padroniza colunas e cria bases processadas por ano.
    "evasao.processamento_dados.pre_processamento",
    # Consolida série temporal e gera dataset temporal sem leakage.
    "evasao.processamento_dados.tratar_dados",
    # Produz análises exploratórias descritivas.
    "evasao.analises.analises",
    # Treina baselines, Regressão Linear e Random Forest no holdout temporal.
    # Gera comparacao_baselines_temporal.*, diagnosticos e feature importance.
    "evasao.modelagem.randomforest",
    # Gera gráficos finais de visualização.
    "evasao.visualizacao.gerar_graficos",
]

ETAPAS_ACESSORIAS = [
    # Valida baseline linear temporal e gráficos de erro.
    "evasao.analises.validar_modelos_temporais",
    # Executa ablação sem concluintes_t e validação rolling/expanding.
    "evasao.analises.experimentos_temporais_complementares",
]


def executar_modulos(etapas: list[str], titulo: str) -> None:
    print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")
    for etapa in etapas:
        run_module(etapa)


def executar_testes() -> None:
    print(f"\n{'=' * 72}\nTESTES E COMPILACAO\n{'=' * 72}")
    run_command(
        [sys.executable, "-m", "compileall", "src", "tests"],
        label="compileall src tests",
    )
    run_command([sys.executable, "-m", "pytest", "-q"], label="pytest -q")


def perguntar_acessorios() -> bool:
    if not sys.stdin.isatty():
        print(
            "\nExecucao nao interativa detectada. "
            "Pipelines acessorios nao serao executados automaticamente."
        )
        return False

    resposta = input(
        "\nDeseja executar agora os pipelines acessorios "
        "(validacoes temporais, ablacões e testes)? [s/N] "
    )
    return resposta.strip().lower() in {"s", "sim", "y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestra o pipeline principal e as validacoes acessorias do AppEvasao."
    )
    parser.add_argument(
        "--coleta",
        action="store_true",
        help="Executa tambem etapas de coleta/download dos microdados.",
    )
    parser.add_argument(
        "--acessorios",
        action="store_true",
        help="Executa validacoes temporais, ablacões e testes apos o pipeline principal.",
    )
    parser.add_argument(
        "--somente-acessorios",
        action="store_true",
        help="Executa apenas validacoes temporais, ablacões e testes.",
    )
    parser.add_argument(
        "--sem-pergunta",
        action="store_true",
        help="Nao pergunta ao final se deve executar os pipelines acessorios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.somente_acessorios:
        executar_modulos(ETAPAS_ACESSORIAS, "PIPELINES ACESSORIOS")
        executar_testes()
        return

    print(
        "Aviso: por padrao, este orquestrador executa apenas o pipeline principal "
        "com dados locais. Use --coleta para incluir coleta/download dos microdados."
    )

    if args.coleta:
        executar_modulos(ETAPAS_COLETA, "COLETA DE DADOS")

    executar_modulos(ETAPAS_PIPELINE_PRINCIPAL, "PIPELINE PRINCIPAL")

    if args.acessorios or (not args.sem_pergunta and perguntar_acessorios()):
        executar_modulos(ETAPAS_ACESSORIAS, "PIPELINES ACESSORIOS")
        executar_testes()

    # Dashboard permanece manual para preservar a experiência Streamlit.
    # streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py


if __name__ == "__main__":
    main()
