from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DADOS_BRUTO = ROOT / "dados" / "bruto"
DADOS_INTERMEDIARIO = ROOT / "dados" / "intermediario"
DADOS_PROCESSADO = ROOT / "dados" / "processado"

MODELOS = ROOT / "modelos"
EXPERIMENTOS = ROOT / "experimentos"

ARQUIVOS_PY = [
    ROOT / "src" / "evasao" / "analises" / "analises.py",
    ROOT / "src" / "evasao" / "analises" / "comparar_modelos.py",
    ROOT / "src" / "evasao" / "analises" / "prever_com_modelos.py",
    ROOT / "src" / "evasao" / "analises" / "validar_modelos_temporais.py",
    ROOT / "src" / "evasao" / "auditoria" / "auditar_pipeline.py",
    ROOT / "src" / "evasao" / "coleta_dados" / "coleta_dados_oficiais.py",
    ROOT / "src" / "evasao" / "coleta_dados" / "coletar_links_inep.py",
    ROOT / "src" / "evasao" / "dashboard" / "app_evasao.py",
    ROOT / "src" / "evasao" / "modelagem" / "randomforest.py",
    ROOT / "src" / "evasao" / "processamento_dados" / "pre_processamento.py",
    ROOT / "src" / "evasao" / "processamento_dados" / "tratar_dados.py",
    ROOT / "src" / "evasao" / "relatorio" / "apendices_scripts.py",
    ROOT / "src" / "evasao" / "relatorio" / "gerar_relatorio_pdf.py",
    ROOT / "src" / "evasao" / "rodar_pipeline.py",
    ROOT / "src" / "evasao" / "utils" / "inspecao.py",
    ROOT / "src" / "evasao" / "utils" / "paths.py",
    ROOT / "src" / "evasao" / "visualizacao" / "comparar_predicoes_cursos.py",
    ROOT / "src" / "evasao" / "visualizacao" / "gerar_graficos.py",
]
