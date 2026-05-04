# Metodologia Temporal e Anti-Leakage

## Problema Original

A formulação original tinha risco de data leakage porque a taxa de evasão podia ser
reconstruída diretamente a partir de variáveis contemporâneas, especialmente
ingressantes, concluintes e taxa de conclusão.

## Formulação Atual

O dataset atual usa variáveis disponíveis no ano-base t para prever
`taxa_evasao_alvo` no ano t+1. O holdout temporal é fixo: treino com
`ano_base <= 2018` e teste com `ano_base > 2018`.

## Variáveis Proibidas

`taxa_conclusao_alvo`, `ingressantes_alvo`, `concluintes_alvo` e quaisquer
variáveis futuras ou derivadas diretamente do alvo não podem entrar em X.

## Papel dos Baselines

Os baselines tornam a comparação auditável:

- média histórica do treino;
- persistência (`taxa_evasao_t`);
- Regressão Linear;
- Random Forest.

O Random Forest só deve ser defendido cientificamente quando superar esses
baselines no mesmo holdout temporal.
