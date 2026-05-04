# Script: apendices_scripts.py
# Gera um documento Word com todos os códigos-fonte dos scripts principais do projeto, com enumeração de linhas.

from pathlib import Path

from docx import Document
from docx.shared import Pt

BASE_DIR = Path(__file__).resolve().parents[3]

# Lista de caminhos absolutos dos scripts
script_paths = [
    BASE_DIR / "src" / "evasao" / "analises" / "analises.py",
    BASE_DIR / "src" / "evasao" / "analises" / "comparar_modelos.py",
    BASE_DIR / "src" / "evasao" / "analises" / "prever_com_modelos.py",
    BASE_DIR / "src" / "evasao" / "analises" / "validar_modelos_temporais.py",
    BASE_DIR / "src" / "evasao" / "auditoria" / "auditar_pipeline.py",
    BASE_DIR / "src" / "evasao" / "coleta_dados" / "coleta_dados_oficiais.py",
    BASE_DIR / "src" / "evasao" / "coleta_dados" / "coletar_links_inep.py",
    BASE_DIR / "src" / "evasao" / "dashboard" / "app_evasao.py",
    BASE_DIR / "src" / "evasao" / "modelagem" / "randomforest.py",
    BASE_DIR / "src" / "evasao" / "processamento_dados" / "pre_processamento.py",
    BASE_DIR / "src" / "evasao" / "processamento_dados" / "tratar_dados.py",
    BASE_DIR / "src" / "evasao" / "relatorio" / "apendices_scripts.py",
    BASE_DIR / "src" / "evasao" / "relatorio" / "gerar_relatorio_pdf.py",
    BASE_DIR / "src" / "evasao" / "rodar_pipeline.py",
    BASE_DIR / "src" / "evasao" / "utils" / "inspecao.py",
    BASE_DIR / "src" / "evasao" / "utils" / "paths.py",
    BASE_DIR / "src" / "evasao" / "visualizacao" / "comparar_predicoes_cursos.py",
    BASE_DIR / "src" / "evasao" / "visualizacao" / "gerar_graficos.py",
]

# Criação do documento Word
doc = Document()
doc.add_heading("Apêndice - Códigos Python Utilizados", level=1)

for path in script_paths:
    if path.exists():
        doc.add_page_break()
        doc.add_heading(path.name, level=2)
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(f"{i:04d}: {line.rstrip()}")
            font = run.font
            font.name = "Courier New"
            font.size = Pt(9)
    else:
        doc.add_paragraph(
            f"[AVISO] Arquivo não encontrado: {path}", style="Intense Quote"
        )

# Salvar o documento
output_path = BASE_DIR / "relatorios" / "appendice_scripts_codigo_fonte.docx"
output_path.parent.mkdir(parents=True, exist_ok=True)
doc.save(output_path)
print(f"Documento salvo com sucesso em: {output_path}")
