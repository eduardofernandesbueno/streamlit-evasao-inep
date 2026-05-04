# Dicionário de Dados — AppEvasão INEP

Este documento descreve as principais bases, variáveis e artefatos utilizados no projeto **AppEvasão INEP**, com foco em rastreabilidade, reprodutibilidade e prevenção de vazamento de informação (*data leakage*).

## 1. Papel metodológico das bases

O projeto utiliza duas camadas principais de dados:

1. **Base descritiva / EDA**  
   Arquivo: `dados/processado/dados_ingresso_evasao_conclusao.csv`  
   Finalidade: análise exploratória, gráficos, contextualização e indicadores descritivos.

2. **Base temporal preditiva / SBIE-RBIE**  
   Arquivo: `dados/processado/dados_modelagem_temporal_sbie.csv`  
   Finalidade: modelagem temporal anti-leakage, com variáveis observadas no ano-base `t` para prever `taxa_evasao_alvo` no ano-alvo `t+1`.

A base descritiva contém variáveis contemporâneas e taxas calculadas diretamente. Ela **não deve ser usada como matriz X principal da modelagem preditiva**. A base temporal é o centro metodológico do pipeline preditivo.

---

## 2. Base descritiva: `dados_ingresso_evasao_conclusao.csv`

| Coluna | Tipo esperado | Descrição | Uso recomendado | Observação metodológica |
|---|---:|---|---|---|
| `id_ies` | inteiro | Identificador da Instituição de Ensino Superior. | EDA, agrupamentos, filtros. | Não identifica pessoa física. |
| `id_curso` | inteiro | Identificador do curso. | EDA, agrupamentos, séries por curso. | Chave de curso. |
| `nome_curso` | texto | Nome do curso de graduação. | Filtros e visualizações. | Pode exigir padronização textual. |
| `modalidade_ensino` | inteiro/categórico | Modalidade do curso, conforme codificação do INEP. | Estratificação e análise descritiva. | Recomenda-se mapear códigos para rótulos legíveis. |
| `numero_cursos` | numérico | Quantidade agregada de cursos na observação. | EDA e agregações. | Variável estrutural. |
| `vagas_totais` | numérico | Total de vagas ofertadas. | EDA e possível feature temporal em `t`. | Usar com sufixo `_t` na base temporal. |
| `inscritos_totais` | numérico | Total de inscritos. | EDA e possível feature temporal em `t`. | Usar com sufixo `_t` na base temporal. |
| `ingressantes` | numérico | Total de ingressantes no curso/ano. | EDA. | Pode induzir cálculo determinístico se usado no mesmo ano do alvo. |
| `matriculados` | numérico | Total de matriculados no curso/ano. | EDA. | Pode estar relacionado ao desfecho contemporâneo. |
| `concluintes` | numérico | Total de concluintes no curso/ano. | EDA. | Variável sensível para leakage se usada no mesmo ano do alvo. |
| `ano` | inteiro | Ano de referência. | Séries temporais e filtros. | Base para construção de `ano_base` e `ano_alvo`. |
| `taxa_ingresso` | numérico | Indicador derivado de ingresso. | Apenas EDA/descritivo. | Não usar como feature principal sem análise de leakage. |
| `taxa_conclusao` | numérico | Taxa derivada de concluintes/ingressantes. | Apenas EDA/descritivo. | Variável diretamente associada à fórmula da evasão. |
| `taxa_evasao` | numérico | Taxa de evasão contemporânea. | Apenas EDA/descritivo. | Não usar como feature. Na modelagem temporal, o alvo é `taxa_evasao_alvo`. |

### Fórmulas descritivas

As taxas contemporâneas são úteis para análise exploratória, mas podem induzir reconstrução determinística do alvo se usadas incorretamente:

```text
taxa_conclusao = concluintes / ingressantes
taxa_evasao = 1 - taxa_conclusao
```

Por isso, essas variáveis devem ser tratadas como indicadores descritivos, não como entrada direta da modelagem preditiva principal.

---

## 3. Base temporal preditiva: `dados_modelagem_temporal_sbie.csv`

A base temporal reformula o problema como:

```text
features em t -> taxa_evasao_alvo em t+1
```

| Coluna | Tipo esperado | Descrição | Papel no modelo | Observação anti-leakage |
|---|---:|---|---|---|
| `id_curso` | inteiro | Identificador do curso. | Identificação/agrupamento. | Não deve ser usado automaticamente como feature sem justificativa. |
| `nome_curso` | texto | Nome do curso. | Exibição/filtro. | Não usado diretamente no Random Forest principal. |
| `modalidade_ensino` | inteiro/categórico | Modalidade de ensino. | Variável contextual possível. | Se usada como feature, documentar codificação. |
| `id_ies` | inteiro | Identificador da IES. | Identificação/agrupamento. | Não usar como proxy indevido sem análise de viés. |
| `ingressantes_t` | numérico | Ingressantes observados no ano-base `t`. | Feature permitida. | Disponível no ano-base. |
| `concluintes_t` | numérico | Concluintes observados no ano-base `t`. | Feature permitida com ressalva. | Só é válida se a predição ocorrer após fechamento anual de `t`. |
| `matriculados_t` | numérico | Matriculados observados no ano-base `t`. | Feature permitida. | Disponível no ano-base. |
| `vagas_totais_t` | numérico | Vagas ofertadas no ano-base `t`. | Feature permitida. | Variável estrutural de oferta. |
| `inscritos_totais_t` | numérico | Inscritos no ano-base `t`. | Feature permitida. | Variável estrutural de demanda. |
| `ano_base` | inteiro | Ano usado como referência para as features. | Split temporal e auditoria. | Treino: `ano_base <= 2018`; teste: `ano_base > 2018`. |
| `ano_alvo` | inteiro | Ano futuro associado ao alvo. | Auditoria temporal. | Normalmente `ano_alvo = ano_base + 1`. |
| `taxa_evasao_alvo` | numérico | Taxa de evasão no ano-alvo `t+1`. | Target `y`. | Nunca pode entrar em `X`. |

---

## 4. Features ativas no modelo principal

O modelo principal em `src/evasao/modelagem/randomforest.py` utiliza apenas variáveis observadas no ano-base `t`:

```text
ingressantes_t
concluintes_t
matriculados_t
vagas_totais_t
inscritos_totais_t
```

O alvo é:

```text
taxa_evasao_alvo
```

---

## 5. Variáveis proibidas como features

As seguintes variáveis são proibidas na matriz `X` da modelagem preditiva principal:

```text
taxa_evasao_alvo
taxa_conclusao_alvo
taxa_evasao
taxa_conclusao
ingressantes_alvo
concluintes_alvo
matriculados_alvo
qualquer variável futura t+1
```

Essas variáveis são proibidas porque representam o próprio alvo, uma transformação direta do alvo ou informação futura indisponível no momento da predição.

---

## 6. Split temporal

A validação principal utiliza holdout temporal fixo:

| Conjunto | Regra | Finalidade |
|---|---|---|
| Treino | `ano_base <= 2018` | Aprender padrões históricos. |
| Teste | `ano_base > 2018` | Avaliar generalização em anos futuros. |

Essa configuração simula um cenário mais realista de predição futura e reduz o risco de uso de informação do futuro no treinamento.

---

## 7. Baselines comparativos

A avaliação compara quatro abordagens no mesmo conjunto temporal:

| Modelo | Descrição | Papel metodológico |
|---|---|---|
| Média histórica do treino | Prediz a média histórica da evasão do treino. | Baseline mínimo. |
| Persistência `evasao_t` | Usa a evasão histórica recente como previsão. | Baseline temporal simples. |
| Regressão Linear | Modelo estatístico simples. | Baseline paramétrico. |
| Random Forest | Modelo não linear baseado em ensemble de árvores. | Modelo principal. |

O Random Forest só deve ser interpretado como útil se superar baselines simples em holdout temporal.

---

## 8. Diagnóstico de completude temporal

Artefatos gerados:

```text
modelos/resultados_modelos/diagnostico_completude_temporal.csv
modelos/resultados_modelos/diagnostico_completude_temporal.md
```

Esses arquivos registram quantas linhas existem antes e depois do critério de completude temporal por `ano_base`.

Observação importante: o ano-base `2020` pode desaparecer após o `dropna` por ausência completa das features temporais obrigatórias. Isso deve ser tratado como limitação metodológica, não como erro silencioso.

---

## 9. Diagnóstico de balanceamento de classes

Artefatos gerados:

```text
modelos/resultados_modelos/diagnostico_balanceamento_classes.csv
modelos/resultados_modelos/diagnostico_balanceamento_classes.md
```

A classificação binária deriva de `taxa_evasao_alvo` usando threshold padrão de `0.50`.

Como há predominância da classe de alta evasão, a acurácia não deve ser usada isoladamente. Recomenda-se priorizar:

- F1-score;
- precision;
- recall;
- matriz de confusão;
- comparação contra baseline majoritário.

---

## 10. Artefatos de modelagem e interpretação

| Arquivo | Descrição |
|---|---|
| `comparacao_baselines_temporal.csv` | Métricas comparativas entre baselines e Random Forest. |
| `comparacao_baselines_temporal.md` | Versão legível em Markdown da comparação. |
| `feature_importance_random_forest.csv` | Importância das variáveis no Random Forest. |
| `feature_importance_random_forest.md` | Versão interpretável da importância de variáveis. |
| `metodologia_temporal_antileakage.md` | Documentação metodológica do desenho temporal e anti-leakage. |
| `metricas_modelos.txt` | Métricas textuais do modelo principal. |
| `metricas_modelos_subconjuntos.txt` | Métricas por subconjuntos de features. |
| `matriz_confusao.png` | Figura da matriz de confusão. |

---

## 11. Limitações metodológicas conhecidas

- Ausência de variáveis socioeconômicas e exógenas.
- Perda de `ano_base=2020` após critério de completude, quando aplicável.
- `concluintes_t` pressupõe fechamento anual do ano-base.
- Classes binárias desbalanceadas.
- R² moderado, compatível com predição real mais difícil.
- Dados agregados por curso, não por indivíduo.
- Possível heterogeneidade entre áreas, modalidades e instituições.

---

## 12. Regra de ouro para o artigo

A formulação correta do projeto é:

```text
Variáveis observadas no ano-base t são usadas para prever a taxa de evasão no ano-alvo t+1, com validação em holdout temporal e comparação contra baselines simples.
```

Não apresentar a base descritiva contemporânea como evidência principal de performance preditiva.
