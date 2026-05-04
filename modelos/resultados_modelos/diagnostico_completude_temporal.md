# Diagnostico de Completude Temporal

Este relatorio contabiliza quantas linhas existiam antes do `dropna`, quantas sobreviveram depois do criterio de completude temporal e qual foi o percentual de perda por `ano_base`.

Nao foi aplicada imputacao automatica. A remocao por incompletude e apenas diagnosticada para apoiar a discussao de ameacas a validade.

| ano_base | antes_dropna | apos_dropna | linhas_perdidas | percentual_perda |
| --- | ---: | ---: | ---: | ---: |
| 2009 | 20291 | 4412 | 15879 | 78.26% |
| 2010 | 21300 | 18360 | 2940 | 13.80% |
| 2011 | 23213 | 19689 | 3524 | 15.18% |
| 2012 | 23762 | 21194 | 2568 | 10.81% |
| 2013 | 24147 | 21599 | 2548 | 10.55% |
| 2014 | 24223 | 21836 | 2387 | 9.85% |
| 2015 | 24738 | 22179 | 2559 | 10.34% |
| 2016 | 25719 | 23144 | 2575 | 10.01% |
| 2017 | 26404 | 23161 | 3243 | 12.28% |
| 2018 | 26743 | 23855 | 2888 | 10.80% |
| 2019 | 29847 | 22330 | 7517 | 25.19% |
| 2020 | 28507 | 0 | 28507 | 100.00% |
| 2021 | 32215 | 27633 | 4582 | 14.22% |
| 2022 | 34163 | 29815 | 4348 | 12.73% |
| 2023 | 34923 | 31550 | 3373 | 9.66% |

## Ano-base 2020

Para `ano_base=2020`, havia 28507 linhas antes do `dropna` e 0 linhas apos o `dropna`, com perda de 100.00%.

Faltantes por coluna obrigatoria em 2020:
- `ingressantes_t`: 28507
- `concluintes_t`: 28507
- `matriculados_t`: 28507
- `vagas_totais_t`: 28507
- `inscritos_totais_t`: 28507
- `taxa_evasao_alvo`: 0
- `ano_alvo`: 0
