# Ablacao sem concluintes_t

Experimento complementar: compara o Random Forest temporal com todas as features permitidas contra uma versao sem `concluintes_t`. O pipeline principal nao foi alterado.

| experimento | modelo | features | usa_concluintes_t | ano_limite_treino | linhas_treino | linhas_teste | threshold_binario | r2 | mae | rmse | mse | accuracy | precision | recall | f1 | cm_tn | cm_fp | cm_fn | cm_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablacao_sem_concluintes_t | Random Forest completo | ingressantes_t,concluintes_t,matriculados_t,vagas_totais_t,inscritos_totais_t | True | 2018 | 199429 | 111328 | 0.500000 | 0.407455 | 0.169667 | 0.219776 | 0.048301 | 0.769034 | 0.813169 | 0.879911 | 0.845225 | 15406 | 16131 | 9582 | 70209 |
| ablacao_sem_concluintes_t | Random Forest sem concluintes_t | ingressantes_t,matriculados_t,vagas_totais_t,inscritos_totais_t | False | 2018 | 199429 | 111328 | 0.500000 | 0.268256 | 0.195106 | 0.244230 | 0.059648 | 0.749551 | 0.788997 | 0.888058 | 0.835601 | 12587 | 18950 | 8932 | 70859 |
