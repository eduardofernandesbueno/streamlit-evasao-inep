# Machine Learning for Higher Education Dropout — Streamlit & INEP/MEC

**Pattern Recognition · Machine Learning · Data Engineering · Random Forest · Educational Data Mining · Interactive Model Evaluation**

**Academic Project:** 2COP507 — Pattern Recognition (Reconhecimento de Padrões)  
**Institution:** State University of Londrina (UEL), Brazil  
**Program:** M.Sc. in Computer Science  
**Project Author:** Eduardo Fernandes Bueno  
**Course Instructors:** Prof. Dr. Bruno B. Zarpelão and Prof. Dr. Sylvio Barbon Jr.  
**Data Source:** Brazilian Higher Education Census — INEP/MEC  
**Research Period:** 2009–2024  
**Project Type:** Academic Machine Learning and Software Engineering Prototype  
**Professional Affiliation:** Research & Development at UEL's Information Technology Office (ATI/UEL)

[Streamlit Application](https://appevasao.streamlit.app/) · [SBIE 2026 Research](https://doi.org/10.5753/sbie.2026.27263) · [Professional Portfolio](https://eduardofernandesbueno.github.io/)

> **Research integrity notice:** This repository originated as a practical Machine Learning coursework project. Its initial evaluation reported near-perfect predictive metrics, but subsequent methodological analysis identified important concerns about target construction, deterministic relationships, and the risk of data leakage. These historical results are not evidence of independently validated prospective prediction performance. A separate temporal benchmark was subsequently developed for SBIE 2026.

---

## 1. Overview

This repository documents a practical Machine Learning project developed for the Pattern Recognition course (2COP507) in the M.Sc. in Computer Science program at the State University of Londrina (UEL), Brazil.

The objective was to design and implement an educational analytics application capable of acquiring official INEP/MEC microdata, preparing datasets, training machine learning models, evaluating predictive outputs, and presenting the results through an interactive Streamlit interface.

The project combines Data Engineering, applied Machine Learning, exploratory data analysis, model evaluation, and interactive software development.

The original implementation investigated Random Forest regression for estimating dropout-related indicators across Brazilian higher education programs.

Its interactive application also explored user-controlled model parameters, visual diagnostics, individual predictions, and batch processing.

The repository subsequently incorporated methodological improvements related to temporal modeling and evaluation.

**This repository represents an academic prototype and experimental development environment, not a production-certified student dropout prediction service.**

---

## 2. Academic Context

**Course:** Pattern Recognition — 2COP507  
**Program:** M.Sc. in Computer Science  
**University:** State University of Londrina (UEL)  
**Project Author:** Eduardo Fernandes Bueno  
**Instructors:** Prof. Dr. Bruno B. Zarpelão and Prof. Dr. Sylvio Barbon Jr.

The practical assignment involved implementing at least one Machine Learning algorithm studied in the course and presenting its behavior through an interactive application.

The resulting project explored:

- Supervised learning and ensemble methods.
- Data acquisition and preprocessing.
- Regression and auxiliary classification.
- Model configuration and hyperparameter experimentation.
- Evaluation metrics and visual diagnostics.
- Interactive predictions.
- Research documentation and experimental limitations.

The original academic report was prepared using the Brazilian Computer Society (SBC) article format.

---

## 3. Project Objectives

### General Objective

Develop an interactive Machine Learning application for exploring and modeling dropout-related indicators in Brazilian higher education using official INEP/MEC data.

### Specific Objectives

1. Automate collection of official census microdata.
2. Organize raw, intermediate, and processed datasets.
3. Clean and standardize heterogeneous educational records.
4. Select and transform explanatory variables.
5. Develop exploratory analytical visualizations.
6. Train and evaluate Random Forest regression models.
7. Compare models with alternative baselines.
8. Explore temporal train/test separation.
9. Experiment with interactive hyperparameter controls.
10. Display regression and auxiliary classification metrics.
11. Provide individual and batch prediction interfaces.
12. Document methodological limitations and improvement opportunities.

Some objectives correspond to the original application, while others reflect subsequent experimental improvements.

---

## 4. Technology Stack

The technologies below are documented in the academic report, previous repository documentation, and associated experiments.

Not every technology is necessarily required by the current application runtime.

### 4.1 Programming and Data Engineering

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data ingestion, preprocessing, transformation and analysis |
| NumPy | Numerical processing and array operations |
| CSV | Structured educational datasets and analytical outputs |
| Python standard library | File and directory management |
| Python virtual environments | Local dependency isolation |
| INEP/MEC Microdata | Official educational data source |

The data engineering workflow was designed to process heterogeneous annual census datasets and generate structured analytical outputs.

### 4.2 Machine Learning and Statistical Evaluation

| Technology / Method | Role |
|---|---|
| scikit-learn | Supervised Machine Learning |
| RandomForestRegressor | Main regression model |
| Linear Regression | Comparative regression baseline |
| Historical Mean | Statistical baseline in subsequent comparisons |
| Temporal Persistence | Time-based comparison baseline |
| Ensemble Learning | Nonlinear modeling through decision-tree ensembles |
| Feature Engineering | Preparation of explanatory variables |
| Temporal Holdout | Chronological train/test evaluation |
| MSE | Mean Squared Error |
| R² | Coefficient of Determination |
| Accuracy | Auxiliary binary evaluation |
| Precision | Auxiliary classification metric |
| Recall | Auxiliary classification metric |
| F1-score | Auxiliary classification metric |
| Confusion Matrix | Visualization of auxiliary binary evaluation |

The original report describes use of Random Forest with 100 estimators and exploratory comparison against Linear Regression.

Later documentation records a revised temporal approach and explicit baseline comparisons.

### 4.3 Interactive Application and Visualization

| Technology | Role |
|---|---|
| Streamlit | Interactive application framework |
| Matplotlib | Statistical charts and diagnostic plots |
| Seaborn | Exploratory statistical visualization |
| Streamlit widgets | Sliders, select boxes and interactive controls |
| CSV upload | Batch prediction inputs |
| CSV download | Exporting generated results |
| Streamlit Community Cloud | Hosted application environment |

The application explored individual prediction, batch prediction, evaluation charts, and parameter controls.

### 4.4 Development and Research Tools

| Tool / Method | Role |
|---|---|
| Git | Version control |
| GitHub | Repository hosting |
| Jupyter Notebook | Optional exploratory experiments |
| SBC Article Template | Academic report formatting |
| Scientific documentation | Methodology and experimental reporting |
| Python scripts | Modular collection, transformation, training and analysis |
| Automated anti-leakage checks | Guards documented for later temporal experiments |

### 4.5 Additional Experimental and Proposed Technologies

The broader project documentation also mentions:

- TensorFlow and Keras as optional neural network dependencies.
- Feature-based modeling and Fine-Tuning as separate experiments.
- Gradient Boosting, XGBoost, LightGBM and CatBoost as possible future comparisons.
- Rolling forecasting origin and nested cross-validation as further evaluation strategies.

These technologies and methods are not all part of the original submitted practical experiment or the current verified application dependencies.

They should not be presented as completed production capabilities.

---

## 5. Data Source

The project uses historical public microdata from the Brazilian Higher Education Census, maintained by INEP/MEC.

**Official source:**

https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior

The academic documentation describes a research period spanning 2009–2024.

Data processing focused on educational variables relevant to admissions, enrollment, completion, institutional activity, and dropout-related indicators.

### Data Engineering Challenges

The original workflow addressed issues including:

- Differences across annual census datasets.
- Inconsistent variable names.
- Missing observations.
- Data type conversion.
- Preparation of course-level records.
- Data normalization and standardization.
- Selection of explanatory variables.
- Generation of intermediate and final datasets.

Raw official data and processed research artifacts should remain clearly distinguished.

---

## 6. Software Architecture

The documented system follows a modular processing structure.

### Conceptual Workflow

**INEP/MEC Sources → Data Collection → Raw Data → Preprocessing → Analytical Datasets → Machine Learning → Evaluation → Streamlit Interface**

### Main Repository Structure

```text
streamlit-evasao-inep/
├── dados/
│   ├── bruto/
│   ├── intermediario/
│   └── processado/
├── modelos/
├── notebooks/
├── src/
│   └── evasao/
│       ├── analises/
│       ├── auditoria/
│       ├── coleta_dados/
│       ├── dashboard/
│       │   └── app_evasao.py
│       ├── modelagem/
│       ├── processamento_dados/
│       ├── relatorio/
│       ├── utils/
│       ├── visualizacao/
│       └── rodar_pipeline.py
├── requirements.txt
└── README.md
```

This structure is taken from the documented project layout. The presence and completeness of individual files in the current branch should be verified before attempting full pipeline reproduction.

### Architecture Components

**Data Collection:** Scripts responsible for identifying official sources, downloading historical microdata and organizing raw files.

**Data Processing:** Scripts responsible for standardization, cleaning, type conversion, aggregation and analytical dataset generation.

**Modeling:** Scripts for Random Forest regression, experimental evaluation and model comparisons.

**Analytics and Visualization:** Scripts for generating plots, diagnostic summaries and model comparison figures.

**Application Layer:** Streamlit dashboard for interactive presentation and experimentation.

**Orchestration:** `rodar_pipeline.py`, documented as a script coordinating the processing workflow.

This is script-based orchestration, not an Apache Airflow deployment.

---

## 7. Data Collection and Preprocessing

### 7.1 Collect Official Data Links

```bash
python src/evasao/coleta_dados/coletar_links_inep.py
```

### 7.2 Download and Organize Microdata

```bash
python src/evasao/coleta_dados/coleta_dados_oficiais.py
```

### 7.3 Preprocess and Standardize Variables

```bash
python src/evasao/processamento_dados/pre_processamento.py
```

### 7.4 Clean and Generate Analytical Datasets

```bash
python src/evasao/processamento_dados/tratar_dados.py
```

### 7.5 Exploratory Data Analysis

```bash
python src/evasao/analises/analises.py
```

These are documented commands from the original project workflow. They are not represented as independently tested commands for the current repository revision.

---

## 8. Original Machine Learning Experiment

### 8.1 Modeling Task

The original coursework investigated prediction of a continuous dropout-related rate using structural course-level variables.

The reported model was based on:

`RandomForestRegressor`

### 8.2 Original Features

The academic report documents explanatory variables involving:

- Total vacancies.
- Total applicants.
- New entrants.
- Enrolled students.
- Graduates.

The initial modeling approach used dropout-related indicators constructed from educational counts.

### 8.3 Original Hyperparameters

The documented main Random Forest configuration included:

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | None |
| `min_samples_leaf` | 1 |

The Streamlit interface also provided controls for exploring different configurations.

### 8.4 Original Evaluation

The submitted academic report described chronological training and testing partitions:

- Training: 2009–2018.
- Testing: 2019–2024.

The experiment also included an auxiliary binary interpretation using a threshold of 0.50 to distinguish higher and lower dropout-related values.

---

## 9. Historical Results and Methodological Reassessment

### 9.1 Originally Reported Results

The submitted academic report documented approximately:

| Metric | Original reported result |
|---|---:|
| R² | 0.9999 |
| MSE | Approximately 0 |
| Binary Accuracy | 0.9997 |
| Binary Precision | 0.9999 |
| Binary Recall | 0.9996 |
| Binary F1-score | 0.9998 |

These values are preserved as historical academic reporting.

**They must not be interpreted as evidence that the system can prospectively predict dropout with approximately 100% accuracy.**

### 9.2 Why the Initial Results Require Caution

The original report acknowledged an almost deterministic relationship between the explanatory variables and the dropout-related target.

Even when directly derived rate columns are removed from the feature matrix, using the original counts from which the target was calculated can allow a model to approximate the target construction formula.

This can generate extremely optimistic metrics without demonstrating reliable prediction of future educational outcomes.

Temporal separation alone does not resolve the issue when the prediction task itself reconstructs a contemporaneous target from related inputs.

### 9.3 Research Integrity

The scientific significance of these findings lies in recognizing the limitations of the original experimental design.

Relevant methodological concerns include:

- Target construction.
- Deterministic feature-target relationships.
- Availability of explanatory variables at prediction time.
- Data leakage.
- Evaluation generalization.
- Interpretation of extremely high predictive metrics.

The original model remains useful as a coursework implementation and experimental learning artifact.

Its results should not be presented as a validated institutional prediction system.

---

## 10. Methodological Evolution — Temporal Benchmark

A subsequent research phase developed a different prediction formulation.

Instead of reconstructing a dropout-related rate from variables observed in the same period, the revised approach investigated prediction of a future target.

### Temporal Formulation

**Features observed at year `t` → dropout target at year `t+1`.**

The revised research used:

```text
ingressantes_t
concluintes_t
matriculados_t
vagas_totais_t
inscritos_totais_t
```

to estimate:

```text
taxa_evasao_alvo
```

for the subsequent year.

### Anti-Leakage Principles

The later benchmark adopted:

- Chronological train/test separation.
- Removal of directly target-derived predictors.
- Explicit comparison of baselines.
- Consistent evaluation datasets.
- Temporal completeness diagnostics.
- Documentation of the operational prediction time.

**Important:** `concluintes_t` is only available for the intended prediction if the annual base period has already been completed and its data are available.

The revised benchmark is a distinct experimental formulation and should not be conflated with the originally submitted coursework evaluation.

---

## 11. Connection to the Published SBIE 2026 Research

The later study was published as:

**"Evasão no Ensino Superior brasileiro: Mapeamento Estruturado da Literatura em ML, EDM e LA, e Benchmark."**

**Authors:** Eduardo Fernandes Bueno, Decio Wey Berti Junior and Jacques Duílio Brancher.

**Venue:** 37th Brazilian Symposium on Computers in Education (SBIE 2026).

**DOI:** https://doi.org/10.5753/sbie.2026.27263

The publication combined a structured review of 88 Brazilian studies with a temporal Machine Learning benchmark using official INEP/MEC data.

### Primary Published Benchmark

The later Random Forest benchmark reported:

| Metric | SBIE 2026 temporal benchmark |
|---|---:|
| R² | 0.3337 |
| MAE | 0.1778 |
| RMSE | 0.2331 |
| MSE | 0.0543 |

These results refer to a different evaluation design from the original coursework experiment.

They must not be substituted into the original academic report as though they were generated by the same experiment.

### Related Public Resources

**SBIE 2026 research artifacts:**

https://github.com/eduardofernandesbueno/sbie2026-evasao-ensino-superior-artefatos

**Public scientific evidence dashboard:**

https://github.com/eduardofernandesbueno/appevasao-sbie-deploy

**Research evidence application:**

https://appevasaosbie.streamlit.app/

The public SBIE deployment is intentionally lightweight and presents fixed research outputs.

By contrast, this original repository documents a broader experimental Machine Learning application, including interactive model-related activities.

---

## 12. Interactive Streamlit Application

The original application was developed as a practical interface for exploring model behavior.

The academic report describes four main functional sections.

### 12.1 About

Introduces the problem, the dataset, the modeling approach and the application.

### 12.2 Individual Prediction

Allows users to provide course-level structural variables and explore the corresponding model output.

### 12.3 Model Evaluation

Provides model evaluation controls and visual evidence, including:

- Hyperparameter adjustments.
- Regression metrics.
- Auxiliary classification metrics.
- Confusion matrix.
- Observed versus predicted values.
- Feature importance.

Some application paths may retrain or reevaluate models during runtime, depending on the selected experiment.

### 12.4 CSV Upload

Supports experimental batch prediction from tabular input and export of the resulting outputs.

### Interpretation Warning

Interactive predictions are educational demonstrations.

They should not be used to make decisions about individual students or to claim institutionally validated dropout risk estimates.

---

## 13. Model Training and Evaluation Scripts

### Train the Documented Random Forest Model

```bash
python src/evasao/modelagem/randomforest.py
```

### Generate Research Figures

```bash
python src/evasao/visualizacao/gerar_graficos.py
```

### Temporal Model Validation

```bash
python src/evasao/analises/validar_modelos_temporais.py
```

### Compare Course-Level Predictions

```bash
python src/evasao/analises/comparar_predicoes_cursos.py
```

These commands reflect the documented workflow.

Actual execution may require historical data files, environment preparation and correction of machine-specific paths.

---

## 14. Interactive Hyperparameter Exploration

The Streamlit application was designed to support practical investigation of Random Forest configuration.

Examples include:

- Number of estimators (`n_estimators`).
- Maximum tree depth (`max_depth`).
- Model configuration options.
- Comparison of evaluation metrics.
- Visual inspection of regression results.
- Auxiliary high/low dropout classification.

The primary objective of these controls is educational experimentation.

Changing hyperparameters can modify training outputs and evaluation metrics.

These runtime results must remain distinct from the fixed, published SBIE 2026 benchmark.

---

## 15. Diagnostics and Research Artifacts

The documented later experimental workflow generates artifacts under:

```text
modelos/resultados_modelos/
```

Relevant filenames include:

```text
comparacao_baselines_temporal.csv
comparacao_baselines_temporal.md
diagnostico_completude_temporal.csv
diagnostico_completude_temporal.md
diagnostico_balanceamento_classes.csv
diagnostico_balanceamento_classes.md
feature_importance_random_forest.csv
feature_importance_random_forest.md
```

The documentation also references anti-leakage methodology files and tests.

These artifacts are intended to support:

- Comparison of baseline models.
- Temporal coverage analysis.
- Dataset completeness inspection.
- Class balance assessment.
- Feature importance analysis.
- Methodological traceability.

Their presence in a particular repository revision should be checked before claiming complete reproducibility.

---

## 16. Running Locally

### 16.1 Requirements

The original documentation specifies Python 3.8 or higher.

A compatible Python environment should be selected based on the actual dependency versions currently specified by the repository.

### 16.2 Clone the Repository

```bash
git clone https://github.com/eduardofernandesbueno/streamlit-evasao-inep.git
cd streamlit-evasao-inep
```

If the repository is private, authorized GitHub access is required.

### 16.3 Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 16.4 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 16.5 Run the Streamlit Application

```bash
streamlit run src/evasao/dashboard/app_evasao.py
```

### 16.6 Run the Documented Orchestration Script

```bash
python src/evasao/rodar_pipeline.py
```

The full pipeline may require official source data downloads, sufficient storage, and restoration of historical dependencies.

**Execution status:** These commands are provided from the documented project structure. Full execution of the current branch has not been independently verified.

---

## 17. Streamlit Deployment

**Documented original application URL:**

https://appevasao.streamlit.app/

**Main application path:**

```text
src/evasao/dashboard/app_evasao.py
```

**Local launch command:**

```bash
streamlit run src/evasao/dashboard/app_evasao.py
```

The original application's public accessibility and current deployment status require separate verification.

A distinct deployment for the SBIE 2026 research evidence is available at:

https://appevasaosbie.streamlit.app/

The two applications serve different documentation and research purposes.

---

## 18. Data Governance and Responsible Use

The project relies on official public educational data.

However, responsible processing requires attention to:

- Data provenance.
- Source licensing and reuse conditions.
- Privacy and confidentiality.
- Consistency across census releases.
- Target validity.
- Data leakage.
- Model generalization.
- Interpretability.
- Institutional accountability.

The project concerns aggregate educational indicators and must not be represented as validated student-level risk prediction.

Predictive outputs should not independently determine academic intervention, access to services, or educational decisions.

---

## 19. Limitations

### Data Quality

- Historical schema differences.
- Missing observations.
- Inconsistent variables.
- Dependencies on official data availability.

### Modeling

- Potentially deterministic target relationships.
- Overly optimistic historical performance metrics.
- Incomplete representation of socioeconomic factors.
- Limited generalization evidence.
- Sensitivity to feature construction and prediction timing.

### Application

- Experimental runtime model configuration.
- Potential environment-specific dependencies.
- Possible large dataset requirements.
- Limited evidence of production hardening.
- Unverified anonymous access to the original hosted application.

### Scientific Scope

This repository does not constitute:

- An official INEP system.
- An ATI/UEL production service.
- A validated individual dropout prediction platform.
- A peer-reviewed publication of the original coursework experiment.
- Evidence of independently reproduced near-perfect forecasting performance.

---

## 20. Future Engineering and Research Work

Potential development directions include:

### Data Engineering

- Formalizing input schemas and data contracts.
- Expanding data quality tests.
- Improving source versioning and lineage.
- Separating raw and processed datasets.
- Improving environment portability.

### Machine Learning

- Revisiting target definitions.
- Strengthening temporal validation.
- Investigating feature availability.
- Comparing additional algorithms.
- Evaluating model uncertainty and generalization.
- Improving leakage detection.

### Software Engineering

- Adding automated unit and integration tests.
- Improving application modularity.
- Introducing reproducible execution configurations.
- Expanding technical documentation.
- Reviewing public deployment security and dependency versions.

### User Experience and Accessibility

- Improving keyboard navigation.
- Adding clearer explanatory text.
- Expanding accessible chart alternatives.
- Testing with assistive technologies.
- Strengthening communication of model limitations.

These are future directions, not completed or independently validated features.

---

## 21. Academic Attribution

**Project:** Practical Machine Learning Project with Streamlit  
**Course:** Pattern Recognition — 2COP507  
**Institution:** State University of Londrina (UEL)  
**Author:** Eduardo Fernandes Bueno

**Course Instructors:**
- Prof. Dr. Bruno B. Zarpelão
- Prof. Dr. Sylvio Barbon Jr.

The instructors are acknowledged for the academic context and course guidance.

They should not automatically be represented as coauthors of the practical implementation or as having endorsed later repository modifications.

The subsequent SBIE 2026 research has a different, explicitly identified authorship and publication record.

---

## 22. Professional Context — ATI/UEL

Eduardo Fernandes Bueno is a Data Engineer, Data Scientist, and M.Sc. candidate in Computer Science at the State University of Londrina.

He contributes to Research & Development activities at the university's Information Technology Office (**ATI/UEL**).

His broader academic and engineering interests include:

- Data Engineering and Data Integration.
- Data Architecture.
- Machine Learning.
- Educational Data Mining.
- Learning Analytics.
- Software Engineering.
- Data Governance.
- Scientific Reproducibility.
- Responsible AI.
- Privacy and regulatory compliance.

His academic background includes Data Science, Information Technology and Law.

### Institutional Clarification

This repository originated as coursework for the UEL Pattern Recognition discipline.

It is not an official UNIDATA repository or an ATI/UEL operational software deliverable.

The ATI/UEL reference provides professional context and should not be interpreted as evidence of institutional deployment, funding, or endorsement.

---

## 23. Academic Report and Citation

**Original report:**

BUENO, Eduardo Fernandes. *Projeto Prático de Aprendizado de Máquina com Streamlit*. Academic coursework report, Pattern Recognition (2COP507), State University of Londrina, Brazil.

This report documents the original experimental design, implementation and historical evaluation.

It should not be cited as a peer-reviewed journal or conference publication.

For the subsequently published temporal benchmark, consult:

BUENO, Eduardo Fernandes; BERTI JUNIOR, Decio Wey; BRANCHER, Jacques Duílio. *Evasão no Ensino Superior brasileiro: Mapeamento Estruturado da Literatura em ML, EDM e LA, e Benchmark*. SBIE 2026, pp. 690–703.

DOI: https://doi.org/10.5753/sbie.2026.27263

---

## 24. License and Contributions

Consult the repository's `LICENCA` file for the applicable terms.

Contributions and technical suggestions may be proposed through GitHub Issues or Pull Requests where repository permissions allow them.

Public accessibility, when enabled, does not automatically establish unrestricted rights to third-party data, academic documents, or research materials.

---

## 25. Professional Links

**Eduardo Fernandes Bueno**

- **GitHub:** https://github.com/eduardofernandesbueno
- **Portfolio:** https://eduardofernandesbueno.github.io/
- **LinkedIn:** https://www.linkedin.com/in/eduardofernandesbueno/
- **ORCID:** https://orcid.org/0009-0000-8620-0211
- **Curriculum Lattes:** http://lattes.cnpq.br/8716619580595259

---

## 26. Project Summary

This repository documents the development and methodological evolution of a Machine Learning coursework project integrating Python, official educational datasets, Random Forest modeling, statistical evaluation and an interactive Streamlit application.

Its engineering contributions include:

**Data Engineering:** Automated collection, preprocessing, organization and preparation of official educational data.

**Machine Learning:** Random Forest regression, exploratory model configuration, baseline comparisons and evaluation.

**Software Engineering:** Modular Python components, orchestration scripts and interactive application development.

**Scientific Methodology:** Critical evaluation of target construction, data leakage, temporal validation and reproducibility.

**Research Communication:** Interactive visualization, experimental documentation and connections to subsequent peer-reviewed research.

A central outcome of the project's evolution was recognizing that very high model performance does not necessarily imply reliable future prediction.

This motivated a more rigorous temporal formulation subsequently documented in SBIE 2026.

---

**UEL — M.Sc. in Computer Science · Pattern Recognition (2COP507)**

*Python · Streamlit · Machine Learning · Data Engineering · Educational Analytics*

**Eduardo Fernandes Bueno — Research & Development at ATI/UEL, Brazil.**
