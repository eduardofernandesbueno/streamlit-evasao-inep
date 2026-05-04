# # src/evasao/coleta_dados/coleta_dados_oficiais.py
# # Este script realiza a coleta e processamento de microdados oficiais do INEP e MEC.
# # Objeticve: This script performs the collection of official microdata from INEP and MEC.

# import os
# import sys
# import requests
# import pandas as pd
# from zipfile import ZipFile

# # Ajuste de caminho para permitir importar `scripts.*` ao executar este arquivo diretamente
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from evasao.utils.inspecao import registrar_ambiente, auditar_csv

# def baixar_arquivo(url, destino):
#     """
#     Faz o download de um arquivo a partir de uma URL e salva no destino especificado.
#     """
#     try:
#         resposta = requests.get(url, stream=True)
#         resposta.raise_for_status()  # Verifica se houve erro na resposta
#         with open(destino, 'wb') as arquivo:
#             for chunk in resposta.iter_content(chunk_size=8192):
#                 arquivo.write(chunk)
#         print(f'Download concluído: {destino}')
#     except requests.RequestException as e:
#         print(f'Erro ao baixar {url}: {e}')

# def extrair_arquivo(zip_caminho, destino_pasta):
#     """
#     Extrai um arquivo ZIP para o destino especificado e ajusta estrutura se necessário.
#     """
#     try:
#         with ZipFile(zip_caminho, 'r') as zip_ref:
#             zip_ref.extractall(destino_pasta)
#         print(f'Arquivos extraídos para: {destino_pasta}')

#         # Verificar estrutura do diretório após extração
#         for root, dirs, files in os.walk(destino_pasta):
#             print(f'Conteúdo de {root}: {dirs}, {files}')
#             break  # Apenas primeiro nível para não poluir o log

#     except Exception as e:
#         print(f'Erro ao extrair {zip_caminho}: {e}')

# def normalizar_nomes_arquivos(caminho_pasta, ano):
#     """
#     Renomeia os arquivos de IES para seguir o formato esperado: MICRODADOS_ED_SUP_IES_YEAR.CSV.
#     """
#     for root, _, arquivos in os.walk(caminho_pasta):
#         for arquivo in arquivos:
#             if "CADASTRO_IES" in arquivo.upper() and arquivo.upper().endswith('.CSV'):
#                 caminho_antigo = os.path.join(root, arquivo)
#                 caminho_novo = os.path.join(root, f"MICRODADOS_ED_SUP_IES_{ano}.CSV")
#                 os.rename(caminho_antigo, caminho_novo)
#                 print(f'Renomeado: {caminho_antigo} -> {caminho_novo}')

# def processar_microdados(caminho_pasta):
#     """
#     Processa os arquivos de microdados e retorna um DataFrame consolidado.
#     """
#     # Normaliza os nomes dos arquivos de IES antes de processar
#     # Extraindo o ano a partir do nome da pasta (assumindo que a pasta contenha o ano)
#     ano = os.path.basename(caminho_pasta).split('_')[1] if '_' in os.path.basename(caminho_pasta) else 'NA'
#     normalizar_nomes_arquivos(caminho_pasta, ano)
#     arquivos_csv = [os.path.join(caminho_pasta, f) for f in os.listdir(caminho_pasta) if f.upper().endswith('.CSV')]
#     df_lista = []
#     for arquivo in arquivos_csv:
#         try:
#             df = pd.read_csv(arquivo, sep=';', encoding='latin1')
#             df_lista.append(df)
#         except Exception as e:
#             print(f'Erro ao processar {arquivo}: {e}')
#     if df_lista:
#         return pd.concat(df_lista, ignore_index=True)
#     else:
#         print('Nenhum dado processado.')
#         return pd.DataFrame()

# def baixar_dados_ano(fonte: str, url: str, pasta_dados: str):
#     """
#     Função auxiliar para baixar, extrair, processar e auditar os dados de um ano/fonte.
#     Mantém a mesma lógica já utilizada no laço de main(), mas adiciona auditoria.
#     """
#     zip_caminho = os.path.join(pasta_dados, f'{fonte}.zip')
#     pasta_extracao = os.path.join(pasta_dados, fonte)
#     os.makedirs(pasta_extracao, exist_ok=True)

#     # Registro de ambiente para auditoria
#     registrar_ambiente(etapa="coleta_bruta", contexto=fonte)

#     # Baixar o arquivo
#     print(f'Baixando dados da fonte: {fonte}')
#     baixar_arquivo(url, zip_caminho)

#     # Extrair o arquivo
#     print(f'Extraindo dados da fonte: {fonte}')
#     extrair_arquivo(zip_caminho, pasta_extracao)

#     # Processar os dados
#     print(f'Processando dados da fonte: {fonte}')
#     df = processar_microdados(pasta_extracao)

#     # Salvar o DataFrame consolidado e auditar o CSV gerado
#     if not df.empty:
#         caminho_csv = os.path.join(pasta_dados, f'{fonte}_dados_brutos.csv')
#         df.to_csv(caminho_csv, index=False)
#         print(f'Dados processados e salvos em: {caminho_csv}')

#         # Auditoria: mostra head e informações básicas desse CSV bruto consolidado
#         auditar_csv(caminho_csv, etapa="coleta_bruta", contexto=fonte, n=5)
#     else:
#         print(f'Nenhum dado válido processado para {fonte}.')

# def main():
#     # URLs reais dos microdados
#     urls = {
#         'INEP_2024-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2024.zip',
#         'INEP_2023-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2023.zip',
#         'INEP_2022-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2022.zip',
#         'INEP_2021-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2021.zip',
#         'INEP_2020-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2020.zip',
#         'INEP_2019-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2019.zip',
#         'INEP_2018-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2018.zip',
#         'INEP_2017-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2017.zip',
#         'INEP_2016-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2016.zip',
#         'INEP_2015-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2015.zip',
#         'INEP_2014-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2014.zip',
#         'INEP_2013-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2013.zip',
#         'INEP_2012-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2012.zip',
#         'INEP_2011-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2011.zip',
#         'INEP_2010-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2010.zip',
#         'INEP_2009-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2009.zip',
#         # 'INEP_2008-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2008.zip',
#         # 'INEP_2007-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2007.zip',
#         # 'INEP_2006-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2006.zip',
#         # 'INEP_2005-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2005.zip',
#         # 'INEP_2004-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2004.zip',
#         # 'INEP_2003-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2003.zip',
#         # 'INEP_2002-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2002.zip',
#         # 'INEP_2001-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2001.zip',
#         # 'INEP_2000-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2000.zip',
#         # 'INEP_1999-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1999.zip',
#         # 'INEP_1998-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1998.zip',
#         # 'INEP_1997-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1997.zip',
#         # 'INEP_1996-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1996.zip',
#         # 'INEP_1995-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1995.zip'
#     }

#     # Caminho para salvar os dados
#     pasta_dados = './dados/bruto'
#     os.makedirs(pasta_dados, exist_ok=True)

#     # Listar pastas/arquivos já existentes em pasta_dados
#     existentes = os.listdir(pasta_dados)

#     # Fontes já baixadas podem aparecer como pastas "MICRODADOS_..." ou como CSVs consolidados "<fonte>_dados_brutos.csv"
#     pastas_inep = [nome for nome in existentes if nome.startswith("MICRODADOS_CADASTRO_CURSOS_") and os.path.isdir(os.path.join(pasta_dados, nome))]
#     csvs_consolidados = [nome for nome in existentes if nome.endswith("_dados_brutos.csv")]

#     fontes_csv = [nome.replace("_dados_brutos.csv", "") for nome in csvs_consolidados]
#     fontes_baixadas = sorted(set(pastas_inep + fontes_csv))

#     print(f"Pastas/arquivos já existentes em '{pasta_dados}':")
#     if not fontes_baixadas:
#         print("Nenhum microdado foi encontrado — iniciando download automaticamente.")
#         opcao = "1"  # Baixar sem perguntar
#     else:
#         print("Fontes já baixadas (pastas ou CSVs consolidados):", fontes_baixadas)
#         fontes_pendentes = [fonte for fonte in urls.keys() if fonte not in fontes_baixadas]
#         if fontes_pendentes:
#             print("Fontes ainda não baixadas (pendentes):", fontes_pendentes)
#         else:
#             print("Nenhuma fonte pendente: todos os anos já possuem pasta ou CSV consolidado.")

#         print("Anos/URLs configurados para download:", list(urls.keys()))
#         opcao = input("Deseja baixar novamente os arquivos (todos os anos da lista)? (1 = sim, 2 = não): ").strip()

#     if opcao == "2":
#         print("Download ignorado pelo usuário.")
#         return

#     for fonte, url in urls.items():
#         baixar_dados_ano(fonte, url, pasta_dados)

# if __name__ == '__main__':
#     main()

# src/evasao/coleta_dados/coleta_dados_oficiais.py
# Este script realiza a coleta e processamento de microdados oficiais do INEP e MEC.
# Objeticve: This script performs the collection of official microdata from INEP and MEC.

import hashlib
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile, is_zipfile

import pandas as pd
import requests

# leitura otimizada usada em laboratórios de dados
try:
    import pyarrow.csv as pv

    PYARROW_AVAILABLE = True
except Exception:
    PYARROW_AVAILABLE = False

# Ajuste de caminho para permitir importar `scripts.*` ao executar este arquivo diretamente
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from evasao.utils.inspecao import auditar_csv, registrar_ambiente
from evasao.utils.paths import DADOS_BRUTO


def perguntar_usuario(msg):
    """Input seguro que ignora Ctrl+C acidental."""
    try:
        # ignora SIGINT temporariamente
        original = signal.signal(signal.SIGINT, signal.SIG_IGN)
        valor = input(msg).strip()
        signal.signal(signal.SIGINT, original)
        return valor
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupção detectada → assumindo opção '2' (não baixar novamente).")
        return "2"


def baixar_arquivo(url, destino, tentativas=3):

    for tentativa in range(1, tentativas + 1):
        try:
            print(f"\nTentativa {tentativa}/{tentativas}")

            resposta = requests.get(url, stream=True, timeout=60)
            resposta.raise_for_status()

            total = int(resposta.headers.get("content-length", 0))
            baixado = 0

            sha256 = hashlib.sha256()

            with open(destino, "wb") as arquivo:
                for chunk in resposta.iter_content(chunk_size=8192):
                    if chunk:
                        arquivo.write(chunk)
                        sha256.update(chunk)

                        baixado += len(chunk)

                        if total > 0:
                            progresso = baixado / total * 100
                            barras = int(progresso / 2)

                            print(
                                f"\r[{'█' * barras}{'.' * (50 - barras)}] "
                                f"{progresso:5.1f}% "
                                f"{baixado / 1024 / 1024:6.1f}MB / {total / 1024 / 1024:6.1f}MB",
                                end="",
                            )

            print("\nDownload concluído.")

            checksum = sha256.hexdigest()

            print(f"SHA256: {checksum}")

            return checksum

        except Exception as e:
            print(f"\nErro no download: {e}")

            if tentativa < tentativas:
                print("Tentando novamente em 5 segundos...")
                time.sleep(5)
            else:
                raise


def extrair_arquivo(zip_caminho, destino_pasta):
    """
    Extrai um arquivo ZIP para o destino especificado e ajusta estrutura se necessário.
    """
    try:
        with ZipFile(zip_caminho, "r") as zip_ref:
            zip_ref.extractall(destino_pasta)
        print(f"Arquivos extraídos para: {destino_pasta}")

        for root, dirs, files in os.walk(destino_pasta):
            print(f"Conteúdo de {root}: {dirs}, {files}")
            break

    except Exception as e:
        print(f"Erro ao extrair {zip_caminho}: {e}")


def normalizar_nomes_arquivos(caminho_pasta, ano):
    """
    Renomeia os arquivos de IES para seguir o formato esperado: MICRODADOS_ED_SUP_IES_YEAR.CSV.
    """
    for root, _, arquivos in os.walk(caminho_pasta):
        for arquivo in arquivos:
            if "CADASTRO_IES" in arquivo.upper() and arquivo.upper().endswith(".CSV"):
                caminho_antigo = os.path.join(root, arquivo)
                caminho_novo = os.path.join(root, f"MICRODADOS_ED_SUP_IES_{ano}.CSV")
                os.rename(caminho_antigo, caminho_novo)
                print(f"Renomeado: {caminho_antigo} -> {caminho_novo}")


def processar_microdados(caminho_pasta):
    """
    Processa os arquivos de microdados e retorna um DataFrame consolidado.
    """
    ano = (
        os.path.basename(caminho_pasta).split("_")[1]
        if "_" in os.path.basename(caminho_pasta)
        else "NA"
    )
    normalizar_nomes_arquivos(caminho_pasta, ano)

    arquivos_csv = [
        os.path.join(caminho_pasta, f)
        for f in os.listdir(caminho_pasta)
        if f.upper().endswith(".CSV")
    ]

    df_lista = []
    for arquivo in arquivos_csv:
        try:
            # leitura acelerada para microdados grandes
            if PYARROW_AVAILABLE:
                tabela = pv.read_csv(arquivo)
                df = tabela.to_pandas()
            else:
                chunks = pd.read_csv(
                    arquivo,
                    sep=";",
                    encoding="latin1",
                    chunksize=500000,
                    low_memory=True,
                )
                df = pd.concat(chunks, ignore_index=True)

            df_lista.append(df)
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

    if df_lista:
        return pd.concat(df_lista, ignore_index=True)
    else:
        print("Nenhum dado processado.")
        return pd.DataFrame()


def baixar_dados_ano(fonte: str, url: str, pasta_dados: str):
    """
    Função auxiliar para baixar, extrair, processar e auditar os dados de um ano/fonte.
    """
    zip_caminho = os.path.join(pasta_dados, f"{fonte}.zip")
    pasta_extracao = os.path.join(pasta_dados, fonte)
    os.makedirs(pasta_extracao, exist_ok=True)

    registrar_ambiente(etapa="coleta_bruta", contexto=fonte)

    print(f"Baixando dados da fonte: {fonte}")
    baixar_arquivo(url, zip_caminho)

    print(f"Extraindo dados da fonte: {fonte}")
    extrair_arquivo(zip_caminho, pasta_extracao)

    print(f"Processando dados da fonte: {fonte}")
    df = processar_microdados(pasta_extracao)

    if not df.empty:
        caminho_csv = os.path.join(pasta_dados, f"{fonte}_dados_brutos.csv")
        df.to_csv(caminho_csv, index=False)
        print(f"Dados processados e salvos em: {caminho_csv}")

        auditar_csv(caminho_csv, etapa="coleta_bruta", contexto=fonte, n=5)
    else:
        print(f"Nenhum dado válido processado para {fonte}.")


def main():
    # Caminho para salvar os dados (centralizado em utils.paths)
    pasta_dados = DADOS_BRUTO
    os.makedirs(pasta_dados, exist_ok=True)

    # --- NOVA VERIFICAÇÃO SIMPLES ---
    arquivos_existentes = os.listdir(pasta_dados)

    pastas_inep = sorted(
        [
            f
            for f in arquivos_existentes
            if f.startswith("INEP_") and os.path.isdir(os.path.join(pasta_dados, f))
        ]
    )

    zips_inep = sorted(
        [f for f in arquivos_existentes if f.startswith("INEP_") and f.endswith(".zip")]
    )

    csvs_inep = sorted(
        [f for f in arquivos_existentes if f.endswith("_dados_brutos.csv")]
    )

    arquivos_csv_existentes = pastas_inep + zips_inep + csvs_inep

    if arquivos_csv_existentes:
        print("\nArquivos CSV já encontrados na pasta:", pasta_dados)

        anos = []
        for arquivo in arquivos_csv_existentes:
            print(arquivo)
            partes = arquivo.split("_")
            for parte in partes:
                if parte.isdigit() and len(parte) == 4:
                    anos.append(int(parte))

        if anos:
            print(f"\nDados locais disponíveis de {min(anos)} até {max(anos)}.")
        else:
            print("\nDados disponíveis localmente, mas anos não identificados.")

        print("Os microdados do INEP estão disponíveis publicamente a partir de 1995.")

        opcao = perguntar_usuario(
            "Deseja baixar novamente os arquivos? (1 = sim, 2 = não): "
        )

        if opcao == "2":
            print("Download ignorado pelo usuário.")
            return

    # URLs reais dos microdados
    urls = {
        "INEP_2024-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2024.zip",
        "INEP_2023-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2023.zip",
        "INEP_2022-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2022.zip",
        "INEP_2021-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2021.zip",
        "INEP_2020-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2020.zip",
        "INEP_2019-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2019.zip",
        "INEP_2018-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2018.zip",
        "INEP_2017-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2017.zip",
        "INEP_2016-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2016.zip",
        "INEP_2015-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2015.zip",
        "INEP_2014-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2014.zip",
        "INEP_2013-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2013.zip",
        "INEP_2012-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2012.zip",
        "INEP_2011-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2011.zip",
        "INEP_2010-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2010.zip",
        "INEP_2009-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2009.zip",
        "INEP_2008-MICRODADOS-CENSO": "https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2008.zip",
        # 'INEP_2007-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2007.zip',
        # 'INEP_2006-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2006.zip',
        # 'INEP_2005-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2005.zip',
        # 'INEP_2004-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2004.zip',
        # 'INEP_2003-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2003.zip',
        # 'INEP_2002-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2002.zip',
        # 'INEP_2001-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2001.zip',
        # 'INEP_2000-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2000.zip',
        # 'INEP_1999-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1999.zip',
        # 'INEP_1998-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1998.zip',
        # 'INEP_1997-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1997.zip',
        # 'INEP_1996-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1996.zip',
        # 'INEP_1995-MICRODADOS-CENSO': 'https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_1995.zip'
    }

    # verificação extra de integridade (detecta ZIP faltante)
    print("\nVerificando integridade dos microdados locais...")

    for fonte in urls.keys():
        zip_path = os.path.join(pasta_dados, f"{fonte}.zip")
        pasta_path = os.path.join(pasta_dados, fonte)

        if os.path.isdir(pasta_path) and not os.path.exists(zip_path):
            print(f"⚠ ZIP ausente detectado: {fonte}.zip → será baixado novamente")

    tarefas_download = []

    for fonte, url in urls.items():
        zip_path = os.path.join(pasta_dados, f"{fonte}.zip")
        pasta_path = os.path.join(pasta_dados, fonte)
        csv_path = os.path.join(pasta_dados, f"{fonte}_dados_brutos.csv")

        zip_existe = os.path.exists(zip_path)
        pasta_existe = os.path.isdir(pasta_path)
        csv_existe = os.path.exists(csv_path)

        # Caso 1 — CSV consolidado já existe → nada a fazer
        if csv_existe:
            print(f"{fonte} já possui CSV consolidado → skip download")
            continue

        # Caso 2 — ZIP existe mas pasta não → apenas extrair
        if zip_existe and not pasta_existe:
            if is_zipfile(zip_path):
                print(f"{fonte} ZIP encontrado mas pasta ausente → extraindo")
                extrair_arquivo(zip_path, pasta_path)
                continue
            else:
                print(f"{zip_path} corrompido → removendo e refazendo download")
                os.remove(zip_path)
                tarefas_download.append((fonte, url))
                continue

        # Caso 3 — pasta existe mas ZIP não
        if pasta_existe and not zip_existe:
            # verificar se há CSV real dentro da pasta (estrutura do INEP pode ter subpastas)
            csv_encontrado = False
            for root, _, files in os.walk(pasta_path):
                for f in files:
                    if f.upper().endswith(".CSV"):
                        csv_encontrado = True
                        break
                if csv_encontrado:
                    break

            if not csv_encontrado:
                print(
                    f"{fonte} pasta existe mas nenhum CSV encontrado → refazendo download"
                )
                tarefas_download.append((fonte, url))
                continue

            else:
                print(
                    f"{fonte} CSV encontrado mas ZIP ausente → baixando ZIP para garantir integridade"
                )
                tarefas_download.append((fonte, url))
                continue

        # Caso 4 — ZIP e pasta existem → validar ZIP
        if zip_existe and pasta_existe:
            if not is_zipfile(zip_path):
                print(f"{zip_path} corrompido → removendo")
                os.remove(zip_path)
                tarefas_download.append((fonte, url))
                continue

            if not os.listdir(pasta_path):
                print(f"{fonte} pasta vazia apesar do ZIP → reextraindo")
                extrair_arquivo(zip_path, pasta_path)
                continue

            print(f"{fonte} já existe → skip download")
            continue

        # Caso 5 — nada existe
        print(f"{fonte} não existe → adicionando à fila de download")
        tarefas_download.append((fonte, url))

    if tarefas_download:
        print(f"\nIniciando downloads paralelos ({len(tarefas_download)} fontes)...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(baixar_dados_ano, fonte, url, pasta_dados)
                for fonte, url in tarefas_download
            ]

            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
