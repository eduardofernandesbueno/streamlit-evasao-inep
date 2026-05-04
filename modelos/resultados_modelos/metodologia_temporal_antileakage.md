# Metodologia Temporal e Anti-Leakage

## Problema Original

A formulação original tinha risco de data leakage porque a taxa de evasão podia ser
reconstruída diretamente a partir de variáveis contemporâneas, especialmente
ingressantes, concluintes e taxa de conclusão.

## Formulação Atual

O dataset atual usa variáveis disponíveis no ano-base t para prever
`taxa_evasao_alvo` no ano t+1. O holdout temporal é fixo: treino com
`ano_base <= 2018` e teste com `ano_base > 2018`.

As features `ingressantes_t`, `concluintes_t`, `matriculados_t`,
`vagas_totais_t` e `inscritos_totais_t` pertencem ao ano-base t. O uso de
`concluintes_t` pressupõe cenário operacional após o fechamento anual do
ano-base; para decisões durante o ano corrente, essa variável deve ser removida
ou substituída por variável parcial disponível no momento da decisão.

## Variáveis Proibidas

`taxa_evasao_alvo`, `taxa_conclusao_alvo`, `ingressantes_alvo`,
`concluintes_alvo`, `matriculados_alvo` e quaisquer variáveis futuras ou
derivadas diretamente do alvo não podem entrar em X.

`dados_ingresso_evasao_conclusao.csv` é base descritiva/EDA. A modelagem
preditiva principal usa `dados_modelagem_temporal_sbie.csv`.

## Papel dos Baselines

Os baselines tornam a comparação auditável:

- média histórica do treino;
- persistência (`taxa_evasao_t`);
- Regressão Linear;
- Random Forest.

O Random Forest só deve ser defendido cientificamente quando superar esses
baselines no mesmo holdout temporal.

## Diagnósticos adicionais

O pipeline também salva diagnósticos de completude temporal e balanceamento de
classes. O ano-base 2020 é diagnosticado explicitamente quando desaparece após
o `dropna`; não há imputação automática sem decisão metodológica.
