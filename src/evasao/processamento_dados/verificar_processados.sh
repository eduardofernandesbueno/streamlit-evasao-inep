#!/usr/bin/env bash

cd /Users/eddieferb/streamlit-evasao-inep || exit 1

ok=1

echo "=== VERIFICAÇÃO DOS ARQUIVOS PRINCIPAIS ==="
for ano in $(seq 2009 2024); do
  for tipo in ies cursos; do
    arq="dados/processado/dados_${tipo}_${ano}.csv"
    [ -f "$arq" ] && echo "[OK] $arq" || { echo "[FALTA] $arq"; ok=0; }
  done
done

[ $ok -eq 1 ] && echo "TODOS OS ARQUIVOS PRINCIPAIS ESTÃO PRESENTES"

echo

echo "=== VERIFICAÇÃO DOS SUBCONJUNTOS ==="
for ano in $(seq 2009 2024); do
  for arq in \
    "dados/processado/dados_ies_${ano}_ies_identificacao.csv" \
    "dados/processado/dados_ies_${ano}_ies_administrativo.csv" \
    "dados/processado/dados_ies_${ano}_ies_docentes_estrutura.csv" \
    "dados/processado/dados_ies_${ano}_ies_docentes_perfil.csv" \
    "dados/processado/dados_cursos_${ano}_curso_identificacao.csv" \
    "dados/processado/dados_cursos_${ano}_curso_oferta_demanda.csv" \
    "dados/processado/dados_cursos_${ano}_curso_fluxo_academico.csv" \
    "dados/processado/dados_cursos_${ano}_curso_geral_compacto.csv"
  do
    [ -f "$arq" ] && echo "[OK] $arq" || { echo "[FALTA] $arq"; ok=0; }
  done
done

[ $ok -eq 1 ] && echo "TODOS OS SUBCONJUNTOS ESTÃO PRESENTES"