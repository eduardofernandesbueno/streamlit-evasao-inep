Projeto Prático de Aprendizado de Máquina com Streamlit – Reconhecimento de Padrões

Este repositório apresenta uma aplicação completa para predição da evasão acadêmica em cursos de Instituições de Ensino Superior (IES) públicas utilizando os microdados do Censo da Educação Superior do INEP/MEC.  A aplicação foi desenvolvida no contexto da disciplina Reconhecimento de Padrões (2COP507) e segue as exigências dos professores Dr. Bruno B. Zarpelão e Dr. Sylvio Barbon Jr. para o projeto prático com Streamlit.  Além de demonstrar o pipeline de coleta, processamento e modelagem, a aplicação inclui uma interface interativa em Streamlit com opções de ajuste de hiperparâmetros e visualização de resultados.

Objetivo

Desenvolver uma aplicação prática que incorpore pelo menos um algoritmo de aprendizado de máquina estudado na disciplina (regressão, classificação ou ensembles).  O foco é aplicar técnicas de ciência de dados para estimar a taxa de evasão em cursos presenciais a partir dos microdados do INEP/MEC e disponibilizar os resultados por meio de uma interface amigável.  Nesta implementação utiliza-se Random Forest como modelo principal e baselines temporais explícitos, incluindo média histórica, persistência e Regressão Linear.

Estrutura de pastas/arquivos principal
├── dados/                     # Conjunto de dados utilizados no projeto
│   ├── bruto/                 # Dados brutos (microdados INEP)
│   ├── processado/            # Dados após limpeza e pré-processamento
│   └── intermediario/         # Dados temporários
├── modelos/                   # Modelos treinados e métricas salvas
├── notebooks/                 # Notebooks Jupyter (opcional)
├── src/evasao/                # Pacote Python principal do pipeline
│   ├── analises/              # Análises exploratórias e validações
│   ├── auditoria/             # Inspeções, logging, validação estrutural
│   ├── coleta_dados/          # Coleta e download dos microdados do INEP
│   ├── dashboard/app_evasao.py# Aplicação Streamlit
│   ├── modelagem/             # Baselines temporais, RF e avaliação de modelos
│   ├── processamento_dados/   # Pré-processamento e tratamento dos dados
│   ├── relatorio/             # Geração de relatórios, PDF, apêndices
│   ├── utils/                 # Funções auxiliares (inspeção, helpers)
│   ├── visualizacao/          # Gráficos, heatmaps, comparações
│   └── rodar_pipeline.py      # Script orquestrador do pipeline completo
├── requirements.txt             # Lista de dependências
└── README.md                  # Documento principal do projeto

Dependências
	•	Python 3.8 ou superior
	•	Streamlit￼ para a interface interativa
	•	Bibliotecas de modelagem e visualização: pandas, numpy, scikit‑learn, seaborn, matplotlib
	•	tensorflow/keras (opcional) para redes neurais
	•	Consulte o arquivo requirements.txt para a lista completa.  Instale as dependências com:

python3 -m venv .venv
source /Users/eddieferb/streamlit-evasao-inep/.venv/bin/activate
pip install -r requirements.txt

Pipeline do Projeto

## Pipeline de execução

1. Coletar links oficiais do INEP  
   `python src/evasao/coleta_dados/coletar_links_inep.py`

2. Baixar e organizar os microdados  
   `python src/evasao/coleta_dados/coleta_dados_oficiais.py`

3. Pré-processar e padronizar variáveis  
   `python src/evasao/processamento_dados/pre_processamento.py`

4. Tratar dados e gerar bases finais  
   `python src/evasao/processamento_dados/tratar_dados.py`

5. Análises exploratórias  
   `python src/evasao/analises/analises.py`

6. Comparar baselines temporais e treinar Random Forest  
   `python src/evasao/modelagem/randomforest.py`

7. Gerar gráficos para o relatório  
   `python src/evasao/visualizacao/gerar_graficos.py`

8. Subir dashboard Streamlit  
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py`

⸻

Executar o pipeline completo (Script Orquestrador)

Execute todo o pipeline de ponta a ponta (coleta de dados, pré-processamento, modelagem e visualização) usando o script orquestrador. Certifique-se de que o ambiente virtual esteja ativado e que as dependências do arquivo requirements.txt estejam instaladas antes de executar.

Ative o ambiente virtual a partir de qualquer pasta:
   `source /Users/eddieferb/streamlit-evasao-inep/.venv/bin/activate`

Rode o orchestrator:
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/python /Users/eddieferb/streamlit-evasao-inep/src/evasao/rodar_pipeline.py`

Finalmente, rode manualmente o Streamlit dashboard 
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py`

⸻



1. Coleta dos Microdados

Os microdados oficiais do INEP (2009–2024) são baixados diretamente das fontes públicas por meio do script src/evasao/coleta_dados/coleta_dados_oficiais.py.  Este script cria a estrutura de pastas em dados/bruto/ e armazena os arquivos CSV correspondentes aos censos anuais.  A etapa de coleta deve ser executada uma única vez:

python src/evasao/coleta_dados/coletar_links_inep.py
python src/evasao/coleta_dados/coleta_dados_oficiais.py

2. Pré‑processamento e Análise Exploratória

O pré‑processamento consiste em limpar e padronizar os dados, selecionar apenas os cursos presenciais e consolidar as variáveis de interesse.  Execute o script src/evasao/processamento_dados/pre_processamento.py para unificar os anos e gerar os arquivos tratados em dados/processado/:

python src/evasao/processamento_dados/pre_processamento.py


Esta etapa inclui:
	•	Seleção das colunas relevantes para análise descritiva e modelagem temporal (ex.: vagas_totais, inscritos_totais, ingressantes, matriculados, concluintes).
	•	Correção de valores ausentes e conversão de tipos.
	•	Geração de conjuntos dados_cursos_tratado.csv e dados_ies_tratado.csv para modelagem.
	•	Análise exploratória básica (histogramas, mapas de calor), opcionalmente em Jupyter notebooks.

3. Divisão de Dados e Engenharia de Atributos

Os scripts da pasta src/evasao/processamento_dados permitem preparar conjuntos temporais de entrada. Para o projeto Streamlit, a base atual usa atributos observados no ano t (`ingressantes_t`, `concluintes_t`, `matriculados_t`, `vagas_totais_t`, `inscritos_totais_t`) para estimar o alvo futuro `taxa_evasao_alvo` no ano t+1. As taxas contemporâneas `taxa_conclusao` e `taxa_evasao` são variáveis descritivas e não devem entrar como preditoras da evasão no mesmo ano, pois isso apenas reconstruiria uma fórmula direta.

4. Modelagem

Treine os modelos de machine learning disponíveis executando src/evasao/modelagem/randomforest.py.

O pipeline atual adota validação temporal: as features em t predizem `taxa_evasao_alvo` em t+1, com treino em `ano_base <= 2018` e teste em `ano_base > 2018`. Variáveis futuras ou diretamente derivadas do alvo, como `taxa_conclusao_alvo`, são proibidas como features. O script compara média histórica, persistência (`taxa_evasao_t`), Regressão Linear e Random Forest no mesmo holdout temporal, salvando a tabela de métricas e o ranking de feature importance da RF em modelos/resultados_modelos/.

5. Avaliação e Visualização

Use src/evasao/visualizacao/gerar_graficos.py para gerar curvas de evolução das taxas de ingresso, conclusão e evasão, além de matrizes de confusão e comparativos entre modelos.  As métricas são salvas como PNG em modelos/resultados_modelos/.  Para validação temporal e comparações por curso, execute src/evasao/analises/validar_modelos_temporais.py e src/evasao/analises/comparar_predicoes_cursos.py.

6. Interface Streamlit

A aplicação interativa é implementada em src/evasao/dashboard/app_evasao.py.  Este painel carrega o modelo treinado e permite:
	•	Selecionar o curso (Administração, Direito, Engenharia Civil, Medicina) para análise.
	•	Ajustar hiperparâmetros do modelo Random Forest através de sliders e caixas de seleção (por exemplo, número de árvores n_estimators, profundidade máxima max_depth e critério de divisão).  Estes controles são definidos com st.slider e st.selectbox.
	•	Exibir métricas em tempo real (MSE, R²) e uma matriz de confusão para avaliação da classificação binária (evasão alta vs. baixa).
	•	Plotar gráficos de linha comparando as taxas reais e preditas ao longo do tempo.

Para executar o painel:

/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py

7. Vídeo e Relatório

Para a entrega final, grave um vídeo de até 15 minutos explicando: (1) a motivação do projeto, (2) como o pipeline foi construído, (3) o funcionamento do modelo e (4) a interface Streamlit.  Além disso, crie um artigo curto (até 4 páginas) no template da SBC descrevendo a escolha dos algoritmos, os hiperparâmetros testados (justificando os valores escolhidos) e discutindo os resultados e melhorias futuras.

Obs.: O pipeline documentado para publicação prioriza a comparação temporal auditável entre baselines simples e Random Forest, evitando formulações em que o modelo apenas reconstrói a taxa de evasão a partir de variáveis contemporâneas diretamente relacionadas ao alvo.

Sobre o Reuso dos Modelos

Os modelos Random Forest já treinados nos scripts originais podem servir de baseline, mas recomenda‑se ajustá‑los quando integrados à aplicação Streamlit.  Dois motivos principais justificam o ajuste:
	1.	Hiperparâmetros: os scripts fixam valores como n_estimators=100 e max_depth=None.  Para atender aos requisitos de interação, exponha esses parâmetros no painel para que o usuário possa ajustá‑los e observar o impacto nas métricas.
	2.	Atualização de dados: o MVP descrito no Projeto Prático de Aprendizado de Máquina com Streamlit utilizou inicialmente dados de 2023 e, posteriormente, dados de 2009 a 2024.  Caso você utilize um recorte diferente ou incorpore novas features, é necessário re‑treinar o modelo para refletir essas alterações.  O pipeline automatizado permite refazer o treinamento sempre que novos dados forem incluídos.

Portanto, reutilize a estrutura em src/evasao/modelagem/randomforest.py para treinamento, salvamento de métricas, comparação com baselines e interpretação por feature importance.  Isso garantirá que a aplicação atenda aos critérios de avaliação (interface, implementação do modelo e documentação) estabelecidos pelos professores.

Contribuição e Licença

Este projeto foi desenvolvido por Eduardo Fernandes Bueno como Projeto Prático de Aprendizado de Máquina com Streamlit para a disciplina 2COP507.  Sinta‑se à vontade para clonar o repositório, testar outras combinações de algoritmos ou adicionar novos cursos.  A licença do projeto e informações adicionais podem ser encontradas em LICENCA.
Contribuição, contato e citações

- Autor: Eduardo Fernandes Bueno — desenvolvido como Projeto Prático de Aprendizado de Máquina com Streamlit a disciplina 2COP507 (Reconhecimento de Padrões).
- Orientação/Professores: Prof. Bruno B. Zarpelão e Prof. Sylvio Barbon Jr.
- Contato: veja o histórico de commits ou a seção de colaboradores no repositório para informações de contato e contribuições.

Como citar este repositório
- Eduardo F. Bueno (2025). Projeto Prático de Aprendizado de Máquina com Streamlit — Reconhecimento de Padrões (2COP507). Repositório GitHub. Disponível em: (inserir URL do repositório).

Agradecimentos
- Gratidão ao INEP/MEC pela disponibilização dos microdados e às disciplinas afins que forneceram a base teórica e prática para este projeto.

Contribuições
- Sinta‑se à vontade para clonar o repositório, abrir issues, submeter pull requests com correções, novas features ou inclusão de outros cursos e algoritmos. Consulte CONTRIBUTING.md (se presente) para orientações.

Licença
- Consulte o arquivo LICENCA para os termos de uso e distribuição.

Nota metodológica

A base de modelagem do Streamlit é temporal: usa atributos observados no ano t
para estimar `taxa_evasao_alvo` no ano t+1. A validação usa treino com
`ano_base <= 2018` e teste com `ano_base > 2018`.

A coluna inversa do alvo `taxa_conclusao_alvo` não é persistida em
`dados/processado/dados_modelagem_temporal_sbie.csv` e não deve entrar como
feature. Os testes em `tests/` protegem esse contrato anti-leakage.

A etapa de modelagem salva os artefatos de auditoria em `modelos/resultados_modelos/`:
- `comparacao_baselines_temporal.csv` e `.md`: média histórica, persistência,
  Regressão Linear e Random Forest no mesmo holdout temporal;
- `diagnostico_completude_temporal.csv` e `.md`: contagem antes/depois do
  `dropna` por `ano_base`, incluindo diagnóstico explícito de 2020;
- `diagnostico_balanceamento_classes.csv` e `.md`: proporções das classes baixa
  e alta evasão e baseline majoritário;
- `feature_importance_random_forest.csv` e `.md`: ranking de importância do RF;
- `metodologia_temporal_antileakage.md`: nota curta para revisão metodológica.

Riscos metodológicos mitigados

- Leakage temporal: treino usa `ano_base <= 2018` e teste usa `ano_base > 2018`.
- Alvo determinístico: `taxa_conclusao`, `taxa_evasao` e colunas derivadas do
  alvo não entram como features preditivas.
- Avaliação no mesmo conjunto: baselines e Random Forest usam o mesmo holdout
  temporal e o mesmo `y_test`.
- Ausência de baseline: média histórica, persistência, Regressão Linear e
  Random Forest são comparados lado a lado.

Hipóteses operacionais e limitações

- `dados_ingresso_evasao_conclusao.csv` é base descritiva/EDA; a modelagem
  preditiva usa `dados_modelagem_temporal_sbie.csv`.
- `concluintes_t` só é válido após o fechamento do ano-base t. Para decisões
  durante o ano corrente, deve ser removido ou substituído por variável parcial
  disponível no momento da decisão.
- O modelo ainda não inclui variáveis socioeconômicas ou exógenas.
- `ano_base=2020` pode desaparecer após o critério de completude; isso é
  diagnosticado, não imputado automaticamente.
- As classes binárias podem ser desbalanceadas; acurácia é auxiliar. Priorize
  F1, precisão, recall e matriz de confusão.
- O ganho do RF é moderado, não perfeito, porque agora a tarefa é predição
  temporal real, não reconstrução de fórmula.
- Os dados são agregados por curso, não por indivíduo.

Consulte `docs/metodologia_modelagem_temporal.md` para a nota técnica voltada à
revisão SBIE/RBIE.
