# Diagnostico de Balanceamento das Classes

A classificacao binaria foi derivada da taxa continua usando o threshold `0.50`. Este diagnostico existe para evitar que a acuracia seja interpretada isoladamente em cenarios desbalanceados.

| conjunto | total | baixa_evasao | alta_evasao | prop_baixa | prop_alta | baseline_majoritario |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total_pos_dropna | 310757 | 94384 | 216373 | 30.37% | 69.63% | 69.63% |
| treino | 199429 | 62847 | 136582 | 31.51% | 68.49% | 68.49% |
| teste | 111328 | 31537 | 79791 | 28.33% | 71.67% | 71.67% |

## Interpretacao

Acuracia nao deve ser usada como argumento principal quando uma classe domina o conjunto de teste. Para a discussao academica, priorize F1, precision, recall e a matriz de confusao.
