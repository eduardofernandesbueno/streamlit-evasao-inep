# Metodologia de Modelagem Temporal Anti-Leakage

## Problema metodologico original

A primeira formulacao do AppEvasao tratava `taxa_evasao` como alvo no mesmo ano
em que variaveis como ingressantes, matriculados e concluintes eram usadas como
entrada. Como `taxa_evasao` pode ser calculada diretamente a partir da relacao
entre ingressantes e concluintes, essa configuracao induzia risco de vazamento
de informacao, reconstrucao deterministica do alvo e metricas artificialmente
altas.

## Formulacao atual

A modelagem preditiva principal foi reformulada como problema temporal:

```text
features em t -> taxa_evasao_alvo em t+1
```

A base principal e `dados/processado/dados_modelagem_temporal_sbie.csv`. Ela
usa variaveis observadas no ano-base t:

- `ingressantes_t`;
- `concluintes_t`;
- `matriculados_t`;
- `vagas_totais_t`;
- `inscritos_totais_t`.

O alvo e `taxa_evasao_alvo`, correspondente ao ano-alvo t+1. A validacao usa
holdout temporal fixo:

- treino: `ano_base <= 2018`;
- teste: `ano_base > 2018`.

Todos os baselines e o Random Forest sao avaliados nas mesmas linhas apos o
criterio de completude temporal.

## Papel das bases de dados

`dados_ingresso_evasao_conclusao.csv` e uma base descritiva para EDA, graficos e
contextualizacao. Ela contem taxas contemporaneas, como `taxa_conclusao` e
`taxa_evasao`, e nao deve ser usada como matriz X da modelagem preditiva
principal.

`dados_modelagem_temporal_sbie.csv` e a base preditiva. O centro metodologico do
artigo deve ser `src/evasao/modelagem/randomforest.py`, que compara media
historica, persistencia, Regressao Linear e Random Forest no mesmo holdout
temporal.

## Variaveis proibidas como features

As seguintes colunas nao podem entrar em X:

- `taxa_evasao_alvo`;
- `taxa_conclusao_alvo`;
- `taxa_evasao`;
- `taxa_conclusao`;
- `concluintes_alvo`;
- `ingressantes_alvo`;
- `matriculados_alvo`;
- qualquer variavel futura t+1.

## Hipotese operacional sobre `concluintes_t`

`concluintes_t` e uma variavel valida apenas se o cenario de uso ocorrer apos o
fechamento do ano-base t, quando o numero de concluintes do ano ja esta
disponivel. Assim, a predicao deve ser interpretada como estimativa de evasao
futura para planejamento apos consolidacao anual dos dados.

Se o objetivo for apoiar decisoes durante o ano corrente, `concluintes_t` deve
ser removido ou substituido por uma variavel parcial disponivel naquele momento,
como indicadores intermediarios de matricula, trancamento, abandono parcial ou
acompanhamento academico.

## Baselines

A comparacao metodologica inclui:

- media historica do treino;
- persistencia baseada em `taxa_evasao_t`;
- Regressao Linear;
- Random Forest.

O Random Forest so e justificavel se apresentar ganho sobre esses baselines no
mesmo conjunto de teste temporal. O ganho esperado e moderado, nao perfeito, e
nao deve ser apresentado como reconstrucao deterministica.

## Diagnosticos obrigatorios

O pipeline salva:

- `modelos/resultados_modelos/comparacao_baselines_temporal.csv`;
- `modelos/resultados_modelos/comparacao_baselines_temporal.md`;
- `modelos/resultados_modelos/diagnostico_completude_temporal.csv`;
- `modelos/resultados_modelos/diagnostico_completude_temporal.md`;
- `modelos/resultados_modelos/diagnostico_balanceamento_classes.csv`;
- `modelos/resultados_modelos/diagnostico_balanceamento_classes.md`;
- `modelos/resultados_modelos/feature_importance_random_forest.csv`;
- `modelos/resultados_modelos/feature_importance_random_forest.md`.

O ano-base 2020 deve ser diagnosticado explicitamente quando desaparecer apos o
`dropna`. Nao ha imputacao automatica sem decisao metodologica.

## Riscos metodologicos mitigados

- leakage temporal;
- alvo deterministico no mesmo ano;
- avaliacao no mesmo conjunto de treino;
- ausencia de baseline simples;
- uso de variaveis inversas ou derivadas diretamente do alvo.

## Limitacoes remanescentes

- ausencia de variaveis socioeconomicas e exogenas;
- perda de `ano_base=2020` apos criterio de completude, quando aplicavel;
- `concluintes_t` depende do fechamento anual;
- classes binarias podem ser desbalanceadas, logo acuracia nao deve ser metrica principal;
- R2 moderado, compativel com predicao real mais dificil;
- dados agregados por curso, nao por individuo.
