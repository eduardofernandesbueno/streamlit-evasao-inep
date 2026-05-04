# tratar_dados.py
# Script responsável pelo carregamento, limpeza, tratamento e consolidação
# de dados educacionais extraídos dos microdados do INEP/MEC.

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Garante que a raiz do projeto esteja no sys.path para permitir imports do pacote evasao
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evasao.utils.inspecao import auditar_csv, registrar_ambiente
from evasao.utils.paths import DADOS_INTERMEDIARIO, DADOS_PROCESSADO

registrar_ambiente(etapa="tratar_dados", contexto="início")

# Pasta onde o pre_processamento grava os CSVs padronizados
PASTA_PROCESSADO = str(DADOS_PROCESSADO)


def carregar_dados(caminho_entrada: str) -> pd.DataFrame:
    """
    Carrega um CSV separado por ';' com encoding UTF-8.
    """
    try:
        print(f"Carregando dados de: {caminho_entrada}")
        df = pd.read_csv(
            caminho_entrada,
            sep=";",
            encoding="utf-8",
            low_memory=False,
        )
        print("Colunas disponíveis:", df.columns.tolist())
        return df
    except Exception as e:
        raise ValueError(f"Erro ao carregar os dados: {e}")


def tratar_dados(df: pd.DataFrame, colunas_numericas: list[str] | None = None) -> pd.DataFrame:
    """
    Limpeza básica:
      - remove duplicados
      - remove linhas completamente vazias
      - converte colunas numéricas
      - filtra valores negativos e ingressantes > 0
    """
    df = df.drop_duplicates()
    df = df.dropna()

    if colunas_numericas:
        for col in colunas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Garante que não haja valores negativos nas métricas principais
        for col in ["ingressantes", "concluintes", "vagas_totais", "matriculados", "numero_cursos"]:
            if col in df.columns:
                df = df[df[col] >= 0]

        # Só faz sentido manter cursos com pelo menos um ingressante
        if "ingressantes" in df.columns:
            df = df[df["ingressantes"] > 0]

        # Remove linhas que ficaram com NaN após as conversões
        df = df.dropna()

    return df


def salvar_dados_tratados(df: pd.DataFrame, caminho_saida: str) -> None:
    """
    Salva um DataFrame tratado em CSV (;, UTF-8).
    """
    try:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        df.to_csv(caminho_saida, index=False, sep=";", encoding="utf-8")
        print(f"Dados tratados salvos em: {caminho_saida}")
    except Exception as e:
        raise ValueError(f"Erro ao salvar os dados: {e}")


def pivotar_dados_cursos() -> None:
    """
    Gera uma base em série temporal a partir dos arquivos
    'dados_cursos_tratado_YYYY.csv' já tratados.

    Cria 'dados_cursos_serie_temporal.csv' em PASTA_PROCESSADO.
    """
    arquivos = sorted(Path(PASTA_PROCESSADO).glob("dados_cursos_tratado_*.csv"))
    dfs: list[pd.DataFrame] = []

    for arq in arquivos:
        ano_match = re.search(r"(\d{4})", arq.name)
        if not ano_match:
            continue
        ano = int(ano_match.group(1))
        df = pd.read_csv(arq, sep=";", encoding="utf-8")
        df["ano"] = ano
        dfs.append(df)

    if not dfs:
        print("Nenhum arquivo de cursos tratado encontrado para pivotar.")
        return

    df_geral = pd.concat(dfs, ignore_index=True)

    # Colunas de identificação (quando existirem)
    id_cols = ["id_curso", "nome_curso", "modalidade_ensino", "id_ies"]
    id_cols = [col for col in id_cols if col in df_geral.columns]

    df_pivot = pd.DataFrame()

    # Variáveis que serão levadas para série temporal
    for var in ["ingressantes", "concluintes", "matriculados", "vagas_totais", "inscritos_totais"]:
        if var in df_geral.columns:
            tabela = df_geral.pivot_table(index=id_cols, columns="ano", values=var)
            tabela.columns = [f"{var}_{int(col)}" for col in tabela.columns]
            df_pivot = pd.concat([df_pivot, tabela], axis=1)

    df_final = df_geral[id_cols].drop_duplicates().set_index(id_cols)
    df_final = df_final.join(df_pivot).reset_index()

    caminho_serie = os.path.join(PASTA_PROCESSADO, "dados_cursos_serie_temporal.csv")
    salvar_dados_tratados(df_final, caminho_serie)


def calcular_taxas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula taxas de ingresso, conclusão e evasão:
      - taxa_ingresso = ingressantes / vagas_totais
      - taxa_conclusao = concluintes / ingressantes
      - taxa_evasao = 1 - taxa_conclusao
    """
    df = df.copy()

    for coluna in ["ingressantes", "concluintes", "vagas_totais"]:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Mantém apenas cursos com ingressantes e vagas > 0
    df = df[(df["ingressantes"] > 0) & (df["vagas_totais"] > 0)]

    df["taxa_ingresso"] = df["ingressantes"] / df["vagas_totais"]
    df["taxa_conclusao"] = df["concluintes"] / df["ingressantes"]
    df["taxa_evasao"] = 1 - df["taxa_conclusao"]

    # Filtra taxas fora de [0, 1]
    for col in ["taxa_ingresso", "taxa_conclusao", "taxa_evasao"]:
        df = df[(df[col] >= 0) & (df[col] <= 1)]

    return df


def salvar_taxas_consolidadas() -> None:
    """
    Consolida todos os 'dados_cursos_tratado_YYYY.csv' em um único
    arquivo com taxas calculadas: 'dados_ingresso_evasao_conclusao.csv'.
    """
    pattern = os.path.join(PASTA_PROCESSADO, "dados_cursos_tratado_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("Nenhum arquivo de cursos tratado foi encontrado para consolidar.")
        return

    list_df: list[pd.DataFrame] = []
    for f in files:
        try:
            df_temp = pd.read_csv(f, sep=";", encoding="utf-8")
            list_df.append(df_temp)
        except Exception as e:
            print(f"Erro ao ler o arquivo {f}: {e}")

    if not list_df:
        print("Nenhum dado foi carregado para consolidação.")
        return

    df_consolidado = pd.concat(list_df, ignore_index=True)
    df_consolidado = calcular_taxas(df_consolidado)

    caminho_saida = os.path.join(PASTA_PROCESSADO, "dados_ingresso_evasao_conclusao.csv")
    salvar_dados_tratados(df_consolidado, caminho_saida)

    auditar_csv(
        caminho_saida,
        etapa="consolidado_final",
        contexto="dados_ingresso_evasao_conclusao",
        n=10,
    )
    print(f"[OK] Dados consolidados e taxas salvos em: {caminho_saida}")


def ler_taxas_consolidadas() -> pd.DataFrame:
    """
    Lê o arquivo consolidado de taxas de ingresso, conclusão e evasão.
    """
    caminho = os.path.join(PASTA_PROCESSADO, "dados_ingresso_evasao_conclusao.csv")
    try:
        df = pd.read_csv(caminho, sep=";", encoding="utf-8")
        print("Dados consolidados lidos com sucesso.")
        return df
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo de taxas consolidadas: {e}")


def gerar_dataset_modelagem_temporal_sbie() -> None:
    """
    Gera um dataset de modelagem temporal sem leakage contemporâneo.

    Ideia:
      - usa apenas variáveis do ano t como preditores
      - define como alvo a taxa_evasao do ano t+1
      - não usa concluintes/ingressantes/vagas do mesmo ano do alvo

    Saída:
      - dados/processado/dados_modelagem_temporal_sbie.csv
    """
    caminho_serie = os.path.join(PASTA_PROCESSADO, "dados_cursos_serie_temporal.csv")
    if not os.path.exists(caminho_serie):
        print("Arquivo de série temporal não encontrado. Dataset SBIE não foi gerado.")
        return

    df = pd.read_csv(caminho_serie, sep=";", encoding="utf-8", low_memory=False)

    anos_disponiveis = sorted(
        {
            int(m.group(2))
            for col in df.columns
            if (m := re.match(r"^(ingressantes|concluintes|matriculados|vagas_totais|inscritos_totais)_(\d{4})$", col))
        }
    )

    if len(anos_disponiveis) < 2:
        print("Anos insuficientes para gerar dataset temporal SBIE.")
        return

    id_cols = [col for col in ["id_curso", "nome_curso", "modalidade_ensino", "id_ies"] if col in df.columns]
    linhas_modelo: list[pd.DataFrame] = []

    for ano in anos_disponiveis[:-1]:
        prox_ano = ano + 1

        col_ing_t = f"ingressantes_{ano}"
        col_conc_t = f"concluintes_{ano}"
        col_mat_t = f"matriculados_{ano}"
        col_vagas_t = f"vagas_totais_{ano}"
        col_insc_t = f"inscritos_totais_{ano}"

        col_ing_y = f"ingressantes_{prox_ano}"
        col_conc_y = f"concluintes_{prox_ano}"

        if col_ing_y not in df.columns or col_conc_y not in df.columns:
            continue

        base_cols = id_cols.copy()
        feat_map = {
            col_ing_t: "ingressantes_t",
            col_conc_t: "concluintes_t",
            col_mat_t: "matriculados_t",
            col_vagas_t: "vagas_totais_t",
            col_insc_t: "inscritos_totais_t",
        }

        cols_existentes = base_cols + [c for c in feat_map if c in df.columns] + [col_ing_y, col_conc_y]
        df_ano = df[cols_existentes].copy()
        df_ano = df_ano.rename(columns=feat_map)
        df_ano["ano_base"] = ano
        df_ano["ano_alvo"] = prox_ano

        df_ano["ingressantes_alvo"] = pd.to_numeric(df_ano[col_ing_y], errors="coerce")
        df_ano["concluintes_alvo"] = pd.to_numeric(df_ano[col_conc_y], errors="coerce")

        df_ano = df_ano[(df_ano["ingressantes_alvo"] > 0) & (df_ano["concluintes_alvo"] >= 0)]
        taxa_conclusao_alvo = df_ano["concluintes_alvo"] / df_ano["ingressantes_alvo"]
        df_ano["taxa_evasao_alvo"] = 1 - taxa_conclusao_alvo
        df_ano = df_ano[(df_ano["taxa_evasao_alvo"] >= 0) & (df_ano["taxa_evasao_alvo"] <= 1)]

        # remove colunas contemporâneas do alvo para evitar leakage
        df_ano = df_ano.drop(
            columns=[
                col_ing_y,
                col_conc_y,
                "ingressantes_alvo",
                "concluintes_alvo",
                "taxa_conclusao_alvo",
            ],
            errors="ignore",
        )

        linhas_modelo.append(df_ano)

    if not linhas_modelo:
        print("Nenhuma linha válida para gerar o dataset temporal SBIE.")
        return

    df_modelo = pd.concat(linhas_modelo, ignore_index=True)

    caminho_saida = os.path.join(PASTA_PROCESSADO, "dados_modelagem_temporal_sbie.csv")
    salvar_dados_tratados(df_modelo, caminho_saida)
    auditar_csv(
        caminho_saida,
        etapa="dataset_modelagem_temporal",
        contexto="sbie_sem_leakage",
        n=10,
    )
    print(f"[OK] Dataset temporal SBIE salvo em: {caminho_saida}")


def main(year: int = 2024) -> None:
    """
    Fluxo de tratamento por ano:
      1) Lê dados_ies_YYYY.csv e dados_cursos_YYYY.csv
      2) Aplica tratar_dados (limpeza numérica)
      3) Garante coluna 'ano'
      4) Salva em ./dados/intermediario e ./dados/processado
    """
    caminho_ies = os.path.join(PASTA_PROCESSADO, f"dados_ies_{year}.csv")
    caminho_cursos = os.path.join(PASTA_PROCESSADO, f"dados_cursos_{year}.csv")

    caminho_ies_tratado = str(DADOS_INTERMEDIARIO / f"dados_ies_tratado_{year}.csv")
    caminho_cursos_tratado = str(DADOS_INTERMEDIARIO / f"dados_cursos_tratado_{year}.csv")

    caminho_ies_final = os.path.join(PASTA_PROCESSADO, f"dados_ies_tratado_{year}.csv")
    caminho_cursos_final = os.path.join(PASTA_PROCESSADO, f"dados_cursos_tratado_{year}.csv")

    colunas_numericas_ies = [
        "docentes_total",
        "docentes_exercicio",
        "docentes_feminino",
        "docentes_masculino",
    ]

    # Estas colunas numéricas correspondem ao subconjunto
    # 'curso_geral_compacto' gerado no pre_processamento.py.
    colunas_numericas_cursos = [
        "numero_cursos",
        "vagas_totais",
        "inscritos_totais",
        "ingressantes",
        "matriculados",
        "concluintes",
    ]

    # -------------------------- IES --------------------------
    try:
        df_ies = carregar_dados(caminho_ies)
    except ValueError as e:
        print(e)
        df_ies = pd.DataFrame()

    if not df_ies.empty:
        try:
            df_ies_tratado = tratar_dados(df_ies, colunas_numericas=colunas_numericas_ies)
            if "ano" not in df_ies_tratado.columns:
                df_ies_tratado["ano"] = year
            salvar_dados_tratados(df_ies_tratado, caminho_ies_tratado)
            salvar_dados_tratados(df_ies_tratado, caminho_ies_final)
        except ValueError as e:
            print(f"Erro ao processar dados de IES: {e}")
    else:
        print("Nenhum dado de IES disponível para tratar.")

    # ------------------------- CURSOS ------------------------
    try:
        df_cursos = carregar_dados(caminho_cursos)
    except ValueError as e:
        print(e)
        df_cursos = pd.DataFrame()

    if not df_cursos.empty:
        try:
            df_cursos_tratado = tratar_dados(df_cursos, colunas_numericas=colunas_numericas_cursos)
            if "ano" not in df_cursos_tratado.columns:
                df_cursos_tratado["ano"] = year
            salvar_dados_tratados(df_cursos_tratado, caminho_cursos_tratado)
            auditar_csv(
                caminho_cursos_tratado,
                etapa="tratado_intermediario",
                contexto=f"cursos_{year}",
                n=5,
            )
            salvar_dados_tratados(df_cursos_tratado, caminho_cursos_final)
        except ValueError as e:
            print(f"Ano de {year} Erro ao processar dados de Cursos: {e}")
    else:
        print("Nenhum dado de Cursos disponível para tratar.")


if __name__ == "__main__":
    # Processa todos os anos de 2009 a 2024 (ajuste conforme necessidade)
    for year in range(2009, 2025):
        print(f"\n=== Tratando ano {year} ===")
        main(year)

    # Gera série temporal, consolida taxas e cria dataset temporal sem leakage
    pivotar_dados_cursos()
    salvar_taxas_consolidadas()
    df_taxas = ler_taxas_consolidadas()
    print(df_taxas.head())
    gerar_dataset_modelagem_temporal_sbie()
