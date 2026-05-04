# Importancia das Variaveis - Random Forest Temporal

O ranking abaixo foi calculado com `feature_importances_` do Random Forest treinado no mesmo holdout temporal dos baselines.

Interpretacao curta: maior importancia indica maior participacao da variavel nas divisoes das arvores. Isso nao implica causalidade e deve ser lido como evidencia associativa sobre dados agregados por curso.

Variavel mais importante nesta execucao: `concluintes_t`.

SHAP nao foi executado porque o pacote `shap` nao esta instalado.

| rank | feature | importance |
| ---: | --- | ---: |
| 1 | concluintes_t | 0.426112 |
| 2 | inscritos_totais_t | 0.154223 |
| 3 | matriculados_t | 0.153139 |
| 4 | ingressantes_t | 0.144684 |
| 5 | vagas_totais_t | 0.121843 |
