# src/evasao/auditoria/auditar_pipeline.py

from pathlib import Path
import sys

# BASE aponta para a raiz do projeto (padroes/)
BASE = Path(__file__).resolve().parents[2]  # ajusta se necessário

# Garante que a raiz esteja no sys.path para podermos usar `from utils...`
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from evasao.utils.inspecao import auditar_csv, registrar_ambiente


def auditar_dados_brutos(anos):
    for ano in anos:
        caminho_cursos = BASE / f"dados/brutos/{ano}/dados_cursos_{ano}.csv"
        caminho_ies    = BASE / f"dados/brutos/{ano}/dados_ies_{ano}.csv"
        auditar_csv(caminho_cursos, etapa="coleta_bruta", contexto=f"cursos_{ano}", n=5)
        auditar_csv(caminho_ies,    etapa="coleta_bruta", contexto=f"ies_{ano}", n=5)


def auditar_pre_processado(anos):
    for ano in anos:
        caminho = BASE / f"dados/processado/dados_cursos_{ano}.csv"
        auditar_csv(caminho, etapa="pre_processado", contexto=f"cursos_{ano}", n=5)


def auditar_tratado_intermediario(anos):
    for ano in anos:
        caminho = BASE / f"dados/intermediario/dados_cursos_tratado_{ano}.csv"
        auditar_csv(caminho, etapa="tratado_intermediario", contexto=f"cursos_{ano}", n=5)


def auditar_consolidado_final():
    caminho = BASE / "dados/processado/dados_ingresso_evasao_conclusao.csv"
    auditar_csv(caminho, etapa="consolidado_final", contexto="dados_ingresso_evasao_conclusao", n=10)


if __name__ == "__main__":
    registrar_ambiente(etapa="auditoria_pipeline", contexto="offline")

    anos = list(range(2009, 2025))

    auditar_dados_brutos(anos=[2009, 2015, 2020, 2024])  # amostragem
    auditar_pre_processado(anos=[2009, 2015, 2020, 2024])
    auditar_tratado_intermediario(anos=[2009, 2015, 2020, 2024])
    auditar_consolidado_final()