# # src/evasao/coleta_dados/coletar_links_inep.py
# # Este script coleta os links dos microdados do Censo da Educação Superior do site do INEP (1995-2023).
# # Objettive: This script collects the links of the Higher Education Census microdata from the INEP website (1995-2023).

# import os
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# import requests
# from bs4 import BeautifulSoup
# from evasao.utils.inspecao import registrar_ambiente


# def coletar_links_inep(url_base, palavra_chave):
#     """
#     Coleta os links dos microdados do Censo da Educação Superior do INEP.

#     :param url_base: URL da página do INEP com os dados.
#     :param palavra_chave: Palavra-chave que identifica os links desejados.
#     :return: Dicionário com os anos e os links dos microdados.
#     """
#     resposta = requests.get(url_base)
#     if resposta.status_code != 200:
#         raise Exception(f"Erro ao acessar {url_base}: {resposta.status_code}")

#     soup = BeautifulSoup(resposta.content, 'html.parser')
#     links = soup.find_all('a', href=True)

#     urls = {}
#     for link in links:
#         href = link['href']
#         if palavra_chave in href:
#             # Extrai o ano do link e cria a entrada no dicionário
#             for ano in range(1995, 2024):
#                 if str(ano) in href:
#                     urls[f"INEP_{ano}-MICRODADOS-CENSO"] = href
#                     break
#     return urls

# def salvar_links_em_arquivo(urls, caminho_arquivo):
#     """
#     Salva os links em um arquivo de texto no formato especificado.

#     :param urls: Dicionário com os links e descrições.
#     :param caminho_arquivo: Caminho do arquivo onde os links serão salvos.
#     """
#     os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
#     with open(caminho_arquivo, 'w') as arquivo:
#         for chave, url in urls.items():
#             arquivo.write(f"{chave}: {url}\n")
#     print(f"Links salvos em {caminho_arquivo}")

# def main():
#     url_base = "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior"
#     palavra_chave = "microdados_censo_da_educacao_superior"
#     caminho_arquivo = "./dados/bruto/lista-links.txt"
#     registrar_ambiente(etapa="coletar_links_inep", contexto="início")

#     print("Coletando links do INEP...")
#     urls = coletar_links_inep(url_base, palavra_chave)
#     print(f"Total de links coletados: {len(urls)}")
#     salvar_links_em_arquivo(urls, caminho_arquivo)
#     print("Processo concluído.")

# if __name__ == '__main__':
#     main()

# src/evasao/coleta_dados/coletar_links_inep.py
# Este script coleta os links dos microdados do Censo da Educação Superior do site do INEP (1995-2023).
# Objettive: This script collects the links of the Higher Education Census microdata from the INEP website (1995-2023).

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import requests
from bs4 import BeautifulSoup

from evasao.utils.inspecao import registrar_ambiente
from evasao.utils.paths import DADOS_BRUTO


def coletar_links_inep(url_base, palavra_chave):
    """
    Coleta os links dos microdados do Censo da Educação Superior do INEP.

    :param url_base: URL da página do INEP com os dados.
    :param palavra_chave: Palavra-chave que identifica os links desejados.
    :return: Dicionário com os anos e os links dos microdados.
    """
    resposta = requests.get(url_base)
    if resposta.status_code != 200:
        raise Exception(f"Erro ao acessar {url_base}: {resposta.status_code}")

    soup = BeautifulSoup(resposta.content, "html.parser")
    links = soup.find_all("a", href=True)

    urls = {}
    for link in links:
        href = link["href"]
        if palavra_chave in href:
            # Extrai o ano do link e cria a entrada no dicionário
            for ano in range(1995, 2025):
                if str(ano) in href:
                    urls[f"INEP_{ano}-MICRODADOS-CENSO"] = href
                    break
    return urls


def salvar_links_em_arquivo(urls, caminho_arquivo):
    """
    Salva os links em um arquivo de texto no formato especificado.

    :param urls: Dicionário com os links e descrições.
    :param caminho_arquivo: Caminho do arquivo onde os links serão salvos.
    """
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    with open(caminho_arquivo, "w") as arquivo:
        for chave, url in urls.items():
            arquivo.write(f"{chave}: {url}\n")
    print(f"Links salvos em {caminho_arquivo}")


def main():
    url_base = "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior"
    palavra_chave = "microdados_censo_da_educacao_superior"
    caminho_arquivo = str(DADOS_BRUTO / "lista-links.txt")

    registrar_ambiente(etapa="coletar_links_inep", contexto="início")

    # --- VERIFICA SE O ARQUIVO JÁ EXISTE ---
    if os.path.exists(caminho_arquivo):
        print("\nArquivo de links já encontrado:", caminho_arquivo)

        with open(caminho_arquivo, "r") as arquivo:
            linhas = arquivo.readlines()

        if linhas:
            print("\nLinks atualmente registrados:")
            for linha in linhas:
                print(linha.strip())

            anos = []
            for linha in linhas:
                for ano in range(1995, 2025):
                    if str(ano) in linha:
                        anos.append(ano)

            if anos:
                print(
                    f"\nOs dados disponíveis atualmente vão de {min(anos)} até {max(anos)}."
                )
            else:
                print("\nOs dados disponíveis são a partir de 1995.")

            resposta = (
                input("\nDeseja baixar novamente os links do INEP? (s/n): ")
                .strip()
                .lower()
            )

            if resposta != "s":
                print("Processo interrompido pelo usuário. Mantendo lista existente.")
                return

    print("Coletando links do INEP...")
    urls = coletar_links_inep(url_base, palavra_chave)
    print(f"Total de links coletados: {len(urls)}")
    salvar_links_em_arquivo(urls, caminho_arquivo)
    print("Processo concluído.")


if __name__ == "__main__":
    main()
