# src/evasao/utils/inspecao.py

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime

import pandas as pd


def _banner(titulo: str) -> str:
    linha = "=" * len(titulo)
    return f"\n{linha}\n{titulo}\n{linha}\n"


def auditar_csv(
    caminho: str | Path,
    etapa: str,
    contexto: str = "",
    n: int = 5,
    sep: str | None = None,
    encoding: str | None = None,
) -> None:
    """
    Lê um CSV e imprime um resumo de auditoria:
      - etapa/contexto
      - caminho
      - shape
      - tipos das colunas
      - head(n)
      - contagem de NaN por coluna

    Uso típico: chamar no final de cada etapa do pipeline.
    """
    caminho = Path(caminho)

    titulo = f"AUDITORIA CSV | etapa={etapa}"
    if contexto:
        titulo += f" | contexto={contexto}"
    titulo += f" | arquivo={caminho.name}"

    print(_banner(titulo))

    if not caminho.exists():
        print(f"[WARN] Arquivo não encontrado: {caminho}")
        return

    try:
        df = pd.read_csv(caminho, sep=sep, encoding=encoding)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {caminho}: {e}")
        return

    auditar_df(df, etapa=etapa, contexto=f"{contexto} | {caminho.name}", n=n)


def auditar_df(
    df: pd.DataFrame,
    etapa: str,
    contexto: str = "",
    n: int = 5,
) -> None:
    """
    Auditoria rápida de um DataFrame já carregado na memória.
    """
    titulo = f"AUDITORIA DF | etapa={etapa}"
    if contexto:
        titulo += f" | contexto={contexto}"
    print(_banner(titulo))

    print(f"▶ Shape: {df.shape[0]} linhas x {df.shape[1]} colunas\n")

    print("▶ Colunas:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    print("▶ Tipos das colunas:")
    print(df.dtypes)
    print()

    print(f"▶ Head({n}):")
    print(df.head(n))
    print()

    print("▶ Contagem de valores ausentes (NaN) por coluna:")
    print(df.isna().sum())
    print()


def registrar_ambiente(
    etapa: str,
    contexto: str = "",
) -> None:
    """
    Log simples sobre ambiente / cwd / timestamp.
    Pode ser chamado no início de cada script do pipeline.
    """
    titulo = f"AMBIENTE | etapa={etapa}"
    if contexto:
        titulo += f" | contexto={contexto}"
    print(_banner(titulo))

    print(f"Data/hora: {datetime.now().isoformat(timespec='seconds')}")
    print(f"CWD      : {os.getcwd()}")
    print()

if __name__ == "__main__":
    print("Módulo inspecao.py carregado com sucesso.")
    print("Funções disponíveis:")
    print(" - registrar_ambiente(etapa, contexto)")
    print(" - auditar_df(df, etapa, contexto, n)")
    print(" - auditar_csv(caminho, etapa, contexto, n)")
    print("\nEste módulo é utilitário. Para usá‑lo, importe as funções nos scripts do pipeline.")