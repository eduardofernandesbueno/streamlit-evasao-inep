# # pre_processamento.py - Script para carregar, padronizar e salvar os microdados do INEP/MEC.
# # Processa os dados brutos (IES e Cursos) por ano, realizando limpeza, renomeação e formatação padronizada.

# Projeto Prático de Aprendizado de Máquina com Streamlit
# 2COP507 - Reconhecimento de Padrões
# Prof. Bruno B. Zarpelão / Prof. Sylvio Barbon Jr.
# Este script prepara os dados de entrada para o treinamento de modelos preditivos.
# Ele lê as bases de dados 'brutos', verifica se as colunas esperadas estão presentes,
# seleciona as features relevantes (IES e Cursos) e adiciona a coluna de ano.
# Finalmente, organiza as ~80 colunas em subconjuntos menores (domínios temáticos),
# gerando arquivos derivados específicos por subconjunto para facilitar a análise e modelagem.

import csv
import glob
import os
import re
import sys
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd

# Garante que a pasta raiz do projeto (onde está `utils/`) esteja no sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from evasao.utils.inspecao import auditar_csv, registrar_ambiente
from evasao.utils.paths import DADOS_BRUTO, DADOS_PROCESSADO

# ==========================================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================================


# Diretórios de entrada e saída (ajuste conforme sua estrutura)
PASTA_BRUTO = str(DADOS_BRUTO)
PASTA_PROCESSADO = str(DADOS_PROCESSADO)

registrar_ambiente(etapa="pre_processamento", contexto="início")

# ==========================================================================
# COLUNAS DE INTERESSE E MAPEAMENTOS
# ==========================================================================

# Exemplo de colunas relevantes para a base de IES, conforme dicionário do Censo 2023.
# Você pode ir expandindo essa lista com novas variáveis à medida que precisar.
COLUNAS_IES_RELEVANTES = [
    "CO_IES",
    "NO_IES",
    "TP_REDE",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "QT_DOC_TOTAL",
    "QT_DOC_EXE",
    "QT_DOC_EX_FEMI",
    "QT_DOC_EX_MASC",
    # Adicione outras colunas se precisar (p. ex. QT_TEC_TOTAL, etc.)
]

MAPPING_IES = {
    "CO_IES": "id_ies",
    "NO_IES": "nome_ies",
    "TP_REDE": "tipo_rede",  # 1 = pública, 2 = privada
    "TP_CATEGORIA_ADMINISTRATIVA": "cat_adm",  # 1 = Fed, 2 = Est, etc.
    "QT_DOC_TOTAL": "docentes_total",
    "QT_DOC_EXE": "docentes_exercicio",
    "QT_DOC_EX_FEMI": "docentes_feminino",
    "QT_DOC_EX_MASC": "docentes_masculino",
}

# Exemplo de colunas relevantes para a base de Cursos, conforme dicionário do Censo 2023.
COLUNAS_CURSOS_RELEVANTES = [
    "CO_IES",
    "CO_CURSO",
    "NO_CURSO",
    "TP_MODALIDADE_ENSINO",
    "QT_CURSO",
    "QT_VG_TOTAL",
    "QT_INSCRITO_TOTAL",
    "QT_ING",
    "QT_MAT",
    "QT_CONC",
    "QT_DIPLOMADOS",
    "QT_DIPLO",
    # de 1995 a 2008 os concluíntes eram chamados de diplomados. Adicione outras colunas se quiser
    # (por exemplo QT_ING_FEM, QT_MAT_18_24, etc.).
]

MAPPING_CURSOS = {
    "CO_IES": "id_ies",
    "CO_CURSO": "id_curso",
    "NO_CURSO": "nome_curso",
    "TP_MODALIDADE_ENSINO": "modalidade_ensino",  # 1=Presencial, 2=EAD
    "QT_CURSO": "numero_cursos",
    "QT_VG_TOTAL": "vagas_totais",
    "QT_INSCRITO_TOTAL": "inscritos_totais",
    "QT_ING": "ingressantes",
    "QT_MAT": "matriculados",
    "QT_CONC": "concluintes",
    "QT_DIPLOMADOS": "concluintes",
    "QT_DIPLO": "concluintes",
}

# --------------------------------------------------------------------------
# Definição dos subconjuntos temáticos (domínios) em cima das colunas RENOMEADAS
# --------------------------------------------------------------------------

# Subconjuntos para a base de IES
SUBCONJUNTOS_IES = {
    # Identificação e localização básica
    "ies_identificacao": [
        "id_ies",
        "nome_ies",
        "ano",
    ],
    # Estrutura administrativa / categoria
    "ies_administrativo": [
        "id_ies",
        "ano",
        "cat_adm",
    ],
    # Estrutura do corpo docente
    "ies_docentes_estrutura": [
        "id_ies",
        "ano",
        "docentes_total",
        "docentes_exercicio",
    ],
    # Perfil do corpo docente
    "ies_docentes_perfil": [
        "id_ies",
        "ano",
        "docentes_feminino",
        "docentes_masculino",
    ],
}

# Subconjuntos para a base de Cursos
SUBCONJUNTOS_CURSOS = {
    # Identificação do curso e vínculo com a IES
    "curso_identificacao": [
        "id_ies",
        "id_curso",
        "nome_curso",
        "ano",
    ],
    # Oferta e demanda
    "curso_oferta_demanda": [
        "id_ies",
        "id_curso",
        "ano",
        "modalidade_ensino",
        "numero_cursos",
        "vagas_totais",
        "inscritos_totais",
    ],
    # Fluxo acadêmico (ingresso, permanência, conclusão)
    "curso_fluxo_academico": [
        "id_ies",
        "id_curso",
        "ano",
        "ingressantes",
        "matriculados",
        "concluintes",
    ],
    # Visão geral compacta (útil para modelos rápidos / baseline)
    "curso_geral_compacto": [
        "id_ies",
        "id_curso",
        "ano",
        "modalidade_ensino",
        "vagas_totais",
        "inscritos_totais",
        "ingressantes",
        "matriculados",
        "concluintes",
    ],
}

# ==========================================================================
# FUNÇÕES DE APOIO
# ==========================================================================


def normalizar_conteudo_pipe(conteudo: str) -> str:
    """
    Remove repetições de delimitadores, espaços desnecessários e caracteres indesejados
    para uniformizar o uso do pipe ("|") como delimitador.
    """
    conteudo = re.sub(r"\|{2,}", "|", conteudo)
    conteudo = re.sub(r"\s*\|\s*", "|", conteudo)
    conteudo = conteudo.replace("\r\n", "\n").replace("\r", "\n")
    conteudo = conteudo.replace('"', "")
    return conteudo


def registrar_problemas(arquivo: str, erro: Exception) -> None:
    """
    Registra problemas de leitura em um log.
    """
    with open("log_erros.txt", "a", encoding="utf-8") as log:
        log.write(f"Arquivo: {arquivo}, Erro: {erro}\n")


def carregar_csv(
    caminho_arquivo: str,
    sep: str = ";",
    encoding: str = "latin1",
    year: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carrega um arquivo CSV a partir de um caminho, tratando a normalização dos delimitadores.

    Se o ano for menor ou igual a 2008 e o conteúdo apresentar o delimitador "|", tenta
    normalizar repetições de delimitadores (por exemplo, '||' ou '|||') para que os dados sejam
    lidos corretamente. Se mesmo assim o DataFrame resultar em apenas uma coluna, tenta separar
    manualmente essa coluna usando o delimitador "|".
    """
    try:
        print(f"Lendo arquivo: {caminho_arquivo}")

        if year is not None and year <= 2008:
            with open(caminho_arquivo, encoding=encoding) as f:
                conteudo = f.read()

            if "|" in conteudo:
                print(
                    f"⚠️ Detecção de separadores múltiplos para ano {year}. Normalizando..."
                )
                conteudo = normalizar_conteudo_pipe(conteudo)
                df = pd.read_csv(
                    StringIO(conteudo),
                    sep="|",
                    header=0,
                    engine="python",
                    on_bad_lines="skip",
                )
                if df.shape[1] == 1:
                    print(
                        "⚠️ Apenas uma coluna detectada após normalização. Tentando separar manualmente..."
                    )
                    df = df.iloc[:, 0].str.split("|", expand=True)
                    df.columns = df.iloc[0]
                    df = df[1:]
                return df

        return pd.read_csv(
            caminho_arquivo,
            sep=sep,
            encoding=encoding,
            header=0,
            usecols=usecols,
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception as e:
        print(f"Erro ao carregar {caminho_arquivo}: {e}")
        registrar_problemas(caminho_arquivo, e)
        return pd.DataFrame()


def filtrar_renomear(
    df: pd.DataFrame, colunas_relevantes: list[str], mapping: dict
) -> pd.DataFrame:
    """
    Seleciona apenas as colunas relevantes contidas no DataFrame e as renomeia conforme
    o dicionário de mapeamento fornecido.
    """
    existentes = [c for c in colunas_relevantes if c in df.columns]
    df_filtrado = df[existentes].copy()
    return df_filtrado.rename(columns=mapping)


def corrigir_nome_pasta(caminho_base: str, ano: int) -> str | None:
    """
    Corrige possíveis problemas com caracteres especiais nos nomes das pastas extraídas,
    retornando o caminho completo da pasta que contenha a string "microdados".
    """
    caminho_esperado = os.path.join(caminho_base, f"INEP_{ano}-MICRODADOS-CENSO")
    if os.path.exists(caminho_esperado):
        for pasta in os.listdir(caminho_esperado):
            pasta_corrigida = (
                unicodedata.normalize("NFKD", pasta)
                .encode("ASCII", "ignore")
                .decode("ASCII")
            )
            pasta_corrigida = re.sub(r"[^a-zA-Z0-9_\- ]", "", pasta_corrigida)
            if "microdados" in pasta_corrigida.lower():
                return os.path.join(caminho_esperado, pasta)
    print(f"Aviso: Nenhuma pasta de microdados encontrada para {ano}")
    return None


def corrigir_nome_arquivo(nome_arquivo: str) -> str:
    """
    Corrige caracteres especiais no nome dos arquivos, removendo aqueles inválidos.
    """
    return re.sub(r"[^a-zA-Z0-9_\-\. ]", "", nome_arquivo)


def salvar_subconjuntos(
    df: pd.DataFrame, ano: int, base: str, subconjuntos: dict[str, list[str]]
) -> None:
    """
    Gera e salva arquivos derivados contendo apenas as colunas de cada subconjunto temático.

    Parâmetros
    ----------
    df : DataFrame já renomeado e com a coluna 'ano'
    ano : ano de referência (para compor o nome dos arquivos)
    base : 'ies' ou 'cursos'
    subconjuntos : dicionário {nome_subconjunto: [lista_de_colunas_renomeadas]}
    """
    chunk_size = 50000

    for nome_sub, cols in subconjuntos.items():
        cols_existentes = [c for c in cols if c in df.columns]
        if not cols_existentes:
            continue

        nome_arquivo = os.path.join(
            PASTA_PROCESSADO, f"dados_{base}_{ano}_{nome_sub}.csv"
        )
        os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)

        total_linhas = len(df)
        primeira_escrita = True

        for inicio in range(0, total_linhas, chunk_size):
            fim = min(inicio + chunk_size, total_linhas)
            df_chunk = df.iloc[inicio:fim][cols_existentes]
            df_chunk.to_csv(
                nome_arquivo,
                sep=";",
                index=False,
                encoding="utf-8",
                mode="w" if primeira_escrita else "a",
                header=primeira_escrita,
                quoting=csv.QUOTE_MINIMAL,
            )
            primeira_escrita = False

        print(f"[OK] subconjunto '{nome_sub}' ({base}) gerado em: {nome_arquivo}")


# ==========================================================================
# PROCESSAMENTO PRINCIPAL
# ==========================================================================


def main(year: int = 2024) -> None:
    arquivos_disponiveis: list[str] = []
    caminho_base_ano = corrigir_nome_pasta(PASTA_BRUTO, year)

    if caminho_base_ano is None:
        print(f"Aviso: caminho base para o ano {year} não encontrado.")
        return

    caminho_dados = os.path.join(caminho_base_ano, "dados")

    if os.path.isdir(caminho_dados):
        arquivos_disponiveis = os.listdir(caminho_dados)

    if year < 2009:
        # Renomeia arquivos com padrões diferentes de nomenclatura para padronizá-los
        for arq in arquivos_disponiveis:
            if "INSTITUICAO" in arq.upper():
                os.rename(
                    os.path.join(caminho_dados, arq),
                    os.path.join(caminho_dados, f"MICRODADOS_ED_SUP_IES_{year}.CSV"),
                )
            elif "GRADUACAO_PRESENCIAL" in arq.upper():
                os.rename(
                    os.path.join(caminho_dados, arq),
                    os.path.join(
                        caminho_dados, f"MICRODADOS_CADASTRO_CURSOS_{year}.CSV"
                    ),
                )
        arquivos_disponiveis = os.listdir(caminho_dados)

    arquivo_ies_esperado_moderno = f"MICRODADOS_ED_SUP_IES_{year}.CSV"
    arquivo_ies_esperado_antigo = f"MICRODADOS_CADASTRO_IES_{year}.CSV"
    ARQUIVO_CURSOS = f"MICRODADOS_CADASTRO_CURSOS_{year}.CSV"

    caminho_ies = None
    for arquivo in arquivos_disponiveis:
        nome_normalizado = corrigir_nome_arquivo(arquivo).upper()
        if nome_normalizado in {
            f"MICRODADOS_ED_SUP_IES_{year}.CSV",
            f"MICRODADOS_CADASTRO_IES_{year}.CSV",
        }:
            caminho_ies = os.path.join(caminho_dados, arquivo)
            break

    if caminho_ies is None:
        caminho_ies = os.path.join(caminho_dados, arquivo_ies_esperado_moderno)

    print(f"Arquivos encontrados em {caminho_dados}: {arquivos_disponiveis}")
    print(
        f"Arquivo IES esperado: {arquivo_ies_esperado_moderno} OU {arquivo_ies_esperado_antigo}"
    )
    print(
        f"Arquivo IES identificado: {caminho_ies if os.path.exists(caminho_ies) else 'NÃO ENCONTRADO'}"
    )

    df_ies_final = pd.DataFrame()
    df_cursos_final = pd.DataFrame()

    # ----------------------------------------------------------------------
    # Carregar e processar MICRODADOS_ED_SUP_IES_{year}.CSV
    # ----------------------------------------------------------------------
    if os.path.isfile(caminho_ies):
        df_ies = carregar_csv(
            caminho_ies, year=year, usecols=lambda c: c in COLUNAS_IES_RELEVANTES
        )
        if not df_ies.empty:
            df_ies_final = filtrar_renomear(df_ies, COLUNAS_IES_RELEVANTES, MAPPING_IES)
            # adiciona coluna de ano para possibilitar split temporal posterior
            df_ies_final["ano"] = year
        else:
            print(
                f"Aviso: {arquivo_ies_esperado_moderno} OU {arquivo_ies_esperado_antigo} está vazio ou não pôde ser processado."
            )
    else:
        print("Aviso: Arquivo de IES não encontrado (nem formato moderno nem antigo).")

    # ----------------------------------------------------------------------
    # Carregar e processar MICRODADOS_CADASTRO_CURSOS_{year}.CSV
    # ----------------------------------------------------------------------
    caminho_cursos = os.path.join(caminho_dados, corrigir_nome_arquivo(ARQUIVO_CURSOS))
    if os.path.isfile(caminho_cursos):
        df_cursos = carregar_csv(
            caminho_cursos, year=year, usecols=lambda c: c in COLUNAS_CURSOS_RELEVANTES
        )
        if not df_cursos.empty:
            df_cursos_final = filtrar_renomear(
                df_cursos, COLUNAS_CURSOS_RELEVANTES, MAPPING_CURSOS
            )
            # adiciona coluna de ano para possibilitar split temporal posterior
            df_cursos_final["ano"] = year
        else:
            print(f"Aviso: {ARQUIVO_CURSOS} está vazio ou não pôde ser processado.")
    else:
        print(f"Aviso: Arquivo {ARQUIVO_CURSOS} não encontrado em {caminho_cursos}.")

    # ----------------------------------------------------------------------
    # Salvando resultados (IES e Cursos) na pasta "processado"
    # ----------------------------------------------------------------------
    if not df_ies_final.empty:
        saida_ies = os.path.join(PASTA_PROCESSADO, f"dados_ies_{year}.csv")
        os.makedirs(os.path.dirname(saida_ies), exist_ok=True)
        df_ies_final.to_csv(saida_ies, sep=";", index=False, encoding="utf-8")
        print(f"[OK] dados_ies gerado em: {saida_ies}")
        # auditoria rápida da saída consolidada
        auditar_csv(saida_ies, etapa="pre_processado", contexto=f"ies_{year}", n=5)
        # geração dos subconjuntos temáticos da base de IES
        salvar_subconjuntos(df_ies_final, year, "ies", SUBCONJUNTOS_IES)
    else:
        print("Nenhum dado de IES para salvar.")

    if not df_cursos_final.empty:
        saida_cursos = os.path.join(PASTA_PROCESSADO, f"dados_cursos_{year}.csv")
        os.makedirs(os.path.dirname(saida_cursos), exist_ok=True)
        df_cursos_final.to_csv(saida_cursos, sep=";", index=False, encoding="utf-8")
        print(f"[OK] dados_cursos gerado em: {saida_cursos}")
        # auditoria rápida da saída consolidada
        auditar_csv(
            saida_cursos, etapa="pre_processado", contexto=f"cursos_{year}", n=5
        )
        # geração dos subconjuntos temáticos da base de Cursos
        salvar_subconjuntos(df_cursos_final, year, "cursos", SUBCONJUNTOS_CURSOS)
    else:
        print("Nenhum dado de Cursos para salvar.")


if __name__ == "__main__":
    # Exemplo: processar todos os anos de 2009 a 2024 (ajuste conforme os microdados disponíveis)
    anos = list(range(2009, 2025))
    for i, ano in enumerate(anos, start=1):
        total_anos = len(anos)
        percentual = (i / total_anos) * 100

        print("\n" + "=" * 80)
        print(f"PROCESSANDO ANO {ano} ({i}/{total_anos})")
        print(f"PROGRESSO DO PRÉ-PROCESSAMENTO: {percentual:.1f}%")
        print("=" * 80)
        main(ano)
