Practical Machine Learning Project with Streamlit — Pattern Recognition

This repository presents a complete application for predicting academic dropout in on-campus programs at public Higher Education Institutions (HEIs) using microdata from the INEP/MEC Higher Education Census. The application was developed as part of the Pattern Recognition course (2COP507) and follows the requirements set by Professors Bruno B. Zarpelão and Sylvio Barbon Jr. for the Streamlit practical project. In addition to demonstrating the pipeline for data collection, processing, and modeling, the application includes an interactive Streamlit interface with options for tuning hyperparameters and visualizing results.

Objective

Develop a practical application that incorporates at least one machine learning algorithm studied in the course (regression, classification, or ensembles). The goal is to apply data science techniques to estimate dropout rates in on-campus courses using INEP/MEC microdata and present the results through a user-friendly interface. In this implementation, Random Forest is the main model and is compared with explicit temporal baselines: historical mean, persistence, and Linear Regression.

Main Folder Structure
├── dados/                     # Datasets used in the project
│   ├── bruto/                 # Raw INEP microdata
│   ├── processado/            # Data after cleaning and preprocessing
│   └── intermediario/         # Temporary intermediate data
├── modelos/                   # Trained models and saved evaluation metrics
├── notebooks/                 # Jupyter notebooks (optional)
├── src/evasao/                   # Python scripts for the entire pipeline
│   ├── analises/              # Exploratory analyses and validation routines
│   ├── auditoria/             # Structural checks, logging, inspection tools
│   ├── coleta_dados/          # INEP microdata retrieval and download automation
│   ├── dashboard/             # Streamlit UI layer
│   │   └── app_evasao.py      # Interactive Streamlit dashboard
│   ├── modelagem/             # Model training, RF, Feature-based, evaluation
│   ├── processamento_dados/   # Data preprocessing and preparation
│   ├── relatorio/             # PDF report generation and appendices
│   ├── utils/                 # Helper utilities and shared functions
│   ├── visualizacao/          # Charts, heatmaps, comparative visualizations
│   └── rodar_pipeline.py      # Full pipeline orchestrator script
├── requirements.txt             # Project dependency list
└── README.md                  # Main project documentation

Here is the requested translation into U.S. English. All folder and script names (e.g., requirements.txt, src/evasao/coleta_dados/coleta_dados_oficiais.py) have been preserved in Portuguese as instructed.

⸻

Dependencies
	•	Python 3.8 or higher
	•	Streamlit for the interactive interface
	•	Modeling and visualization libraries: pandas, numpy, scikit-learn, seaborn, matplotlib
	•	tensorflow/keras (optional) for neural networks
	•	See the requirements.txt file for the complete list. Install dependencies with:
python3 -m venv .venv
source /Users/eddieferb/streamlit-evasao-inep/.venv/bin/activate
pip install -r requirements.txt


⸻

Project Pipeline
## Execution Pipeline

1. Collect official INEP links  
   `python src/evasao/coleta_dados/coletar_links_inep.py`

2. Download and organize microdata  
   `python src/evasao/coleta_dados/coleta_dados_oficiais.py`

3. Preprocess and standardize variables  
   `python src/evasao/processamento_dados/pre_processamento.py`

4. Clean data and generate final datasets
   `python src/evasao/processamento_dados/tratar_dados.py`

5. Exploratory data analysis  
   `python src/evasao/analises/analises.py`

6. Train prediction models  
   `python src/evasao/modelagem/randomforest.py`

7. Generate charts for the report
   `python src/evasao/visualizacao/gerar_graficos.py`

8. Launch Streamlit dashboard 
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py`

⸻

Run the end-to-end pipeline (Orchestrator script)

Execute the complete pipeline (data collection, preprocessing, modeling, and visualization) using the orchestrator script. Ensure the virtual environment is activated and dependencies from requirements.txt are installed before running.

Activate the virtual environment from any folder:
   `source /Users/eddieferb/streamlit-evasao-inep/.venv/bin/activate`

Run the orchestrator:
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/python /Users/eddieferb/streamlit-evasao-inep/src/evasao/rodar_pipeline.py`

Then, run manually Launch Streamlit dashboard 
   `/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py`

⸻

1. Microdata Collection
The official INEP microdata (2009–2024) are downloaded directly from public sources using the script src/evasao/coleta_dados/coleta_dados_oficiais.py.
This script creates the folder structure under dados/bruto/ and stores the CSV files corresponding to annual censuses.
This collection step needs to be executed only once:
python src/evasao/coleta_dados/coletar_links_inep.py
python src/evasao/coleta_dados/coleta_dados_oficiais.py

2. Preprocessing and Exploratory Analysis
Preprocessing involves cleaning and standardizing the data, selecting only on-campus programs, and consolidating key variables.
Run the script src/evasao/processamento_dados/pre_processamento.py to unify the years and generate processed files under dados/processado/:
python src/evasao/processamento_dados/pre_processamento.py
python src/evasao/processamento_dados/tratar_dados.py

This step includes:
	•	Selection of relevant columns for descriptive analysis and temporal modeling (e.g. vagas_totais, inscritos_totais, ingressantes, matriculados, concluintes)
	•	Handling of missing values and type conversions
	•	Generation of dados_cursos_tratado.csv and dados_ies_tratado.csv for modeling
	•	Basic exploratory analysis (histograms, heatmaps), optionally via Jupyter notebooks

3. Data Splitting and Feature Engineering
The scripts under the folder src/evasao/processamento allow preparing temporal input sets.
For the Streamlit project, the current modeling dataset uses features observed in year t
(`ingressantes_t`, `concluintes_t`, `matriculados_t`, `vagas_totais_t`, `inscritos_totais_t`)
to estimate the future target `taxa_evasao_alvo` in year t+1. The contemporaneous rates
`taxa_conclusao` and `taxa_evasao` are descriptive variables and must not be used as input
features to predict same-year dropout, because that would reconstruct a direct formula.

4. Modeling
Train the machine learning models by running the modeling scripts:
	•	Temporal baselines and Random Forest — src/evasao/modelagem/randomforest.py

The current modeling pipeline uses temporal validation: features observed in year t are used to predict `taxa_evasao_alvo` in year t+1. The holdout split uses `ano_base <= 2018` for training and `ano_base > 2018` for testing. Variables directly derived from the future target, such as `taxa_conclusao_alvo`, must not enter X.

The script src/evasao/modelagem/randomforest.py compares historical mean, persistence (`taxa_evasao_t`), Linear Regression, and Random Forest on the same temporal holdout. It saves the metrics table and the Random Forest feature importance ranking under modelos/resultados_modelos/.
You can adjust the hyperparameters directly in the scripts or through the Streamlit panel.

5. Evaluation and Visualization
Use src/evasao/visualizacao/gerar_graficos.py to generate evolution curves of enrollment, completion, and dropout rates, as well as confusion matrices and model comparisons.
The metrics are saved as PNG files under modelos/resultados_modelos/.
For temporal validation and course-specific comparisons, run src/evasao/analises/validar_modelos_temporais.py and src/evasao/analises/comparar_predicoes_cursos.py.

6. Streamlit Interface
The interactive application is implemented in src/evasao/dashboard/app_evasao.py.
This panel loads the trained model and allows users to:
	•	Select a course (e.g. Administration, Law, Civil Engineering, Medicine) for analysis
	•	Adjust hyperparameters of the Random Forest model via sliders and drop-down menus (e.g. number of trees n_estimators, maximum depth max_depth, and criterion) — using st.slider and st.selectbox
	•	Display real-time metrics (MSE, R²) and a confusion matrix for binary classification (high vs. low dropout)
	•	Plot line charts comparing actual vs. predicted rates over time

To run the panel:
/Users/eddieferb/streamlit-evasao-inep/.venv/bin/streamlit run /Users/eddieferb/streamlit-evasao-inep/src/evasao/dashboard/app_evasao.py

7. Video and Report
For the final submission:
	•	Record a video (up to 15 minutes) explaining:
(1) project motivation, (2) the constructed pipeline, (3) how the model works, and (4) the Streamlit interface.
	•	Additionally, prepare a short article (up to 4 pages) using the SBC template. The document should describe the choice of algorithms, tested hyperparameters (with justification), and discuss results and future improvements.
Note: In addition to the main pipeline, I explored parallel approaches (fine-tuning and feature-based modeling) purely as additional experimentation. These methods were not included in the final project deliverable.
⸻

About Model Reuse

Previously trained Random Forest models can serve as a baseline, but it is recommended to adjust them when integrating with the Streamlit app.
Two main reasons:
	1.	Hyperparameters: The scripts currently define fixed values such as n_estimators=100 and max_depth=None.
To meet interaction requirements, expose these parameters in the panel for users to adjust and observe the impact on metrics.
	2.	Data updates: The MVP described in the Streamlit project initially used data from 2023, and later expanded to data from 2009 to 2024.
If you use a different data subset or add new features, the model should be retrained to reflect these changes.

Therefore, reuse the structure of the modeling scripts (training and saving functions), but allow hyperparameter adjustments in the code or interface.
This will ensure the application meets the evaluation criteria (interface, model implementation, documentation) set by the professors.

⸻

Contribution and License

This project was developed by Eduardo Fernandes Bueno as the Machine Learning Practical Project with Streamlit (2COP507 – Recognition of Patterns).
Feel free to clone the repository, test other algorithm combinations, or add new courses.
The license and additional details can be found in LICENCA.

Contributions, Contact, and Citation
	•	Author: Eduardo Fernandes Bueno — developed as the Practical ML Project with Streamlit for course 2COP507 (Recognition of Patterns).
	•	Advisors/Professors: Prof. Bruno B. Zarpelão and Prof. Sylvio Barbon Jr.
	•	Contact: See commit history or collaborators section for contribution and contact details.

How to cite this repository:
Eduardo F. Bueno (2025). Machine Learning Practical Project with Streamlit — Recognition of Patterns (2COP507). GitHub repository. Available at: (insert repository URL).

⸻

Acknowledgments

Special thanks to INEP/MEC for providing the microdata and to the related academic courses for supplying the theoretical and practical foundation for this project.

⸻

Contributions

You are welcome to clone the repository, open issues, and submit pull requests with fixes, new features, or additional course models and algorithms.
If available, see CONTRIBUTING.md for guidelines.

⸻

License

Please see the LICENCA file for terms of use and distribution.

Reproducing the experiments

git clone repo
cd streamlit-evasao-inep

python -m venv .venv
source /Users/eddieferb/streamlit-evasao-inep/.venv/bin/activate

pip install -r requirements.txt

/Users/eddieferb/streamlit-evasao-inep/.venv/bin/python /Users/eddieferb/streamlit-evasao-inep/src/evasao/rodar_pipeline.py


---------------------------------
Methodological note

The modeling dataset for the Streamlit application is temporal. It uses features observed
in year t to estimate `taxa_evasao_alvo` in year t+1, with temporal validation based on
`ano_base <= 2018` for training and `ano_base > 2018` for testing.

The target-inverse column `taxa_conclusao_alvo` is intentionally not persisted in
`dados/processado/dados_modelagem_temporal_sbie.csv` and must not be used as a feature.
Anti-leakage tests under `tests/` guard this contract.

The modeling step writes audit artifacts under `modelos/resultados_modelos/`:

- `comparacao_baselines_temporal.csv` and `.md`: historical mean, persistence,
  Linear Regression, and Random Forest compared on the same temporal holdout;
- `diagnostico_completude_temporal.csv` and `.md`: before/after `dropna`
  counts by `ano_base`, including the explicit 2020 diagnostic;
- `diagnostico_balanceamento_classes.csv` and `.md`: low/high dropout class
  proportions and majority-class baseline;
- `feature_importance_random_forest.csv` and `.md`: Random Forest feature ranking;
- `metodologia_temporal_antileakage.md`: short methodological note for review.

---------------------------------
Riscos metodológicos mitigados

- Temporal leakage: training uses `ano_base <= 2018` and testing uses
  `ano_base > 2018`.
- Deterministic target reconstruction: contemporaneous `taxa_conclusao`,
  `taxa_evasao` and target-derived columns are not used as predictive features.
- Evaluation on the same set: all baselines and Random Forest use the same
  temporal holdout and the same `y_test`.
- Missing baseline comparison: historical mean, persistence, Linear Regression
  and Random Forest are reported side by side.

---------------------------------
Operational assumptions and limitations

- `dados_ingresso_evasao_conclusao.csv` is descriptive/EDA; predictive modeling
  centers on `dados_modelagem_temporal_sbie.csv`.
- `concluintes_t` is valid only after the year-base t has been closed. For
  decisions during the current year, it should be removed or replaced by a
  partial variable available at decision time.
- The model does not include socioeconomic or exogenous variables.
- `ano_base=2020` may be lost after the completeness criterion; this is
  diagnosed, not automatically imputed.
- Binary classes can be imbalanced, so accuracy is auxiliary. Prefer F1,
  precision, recall and confusion matrix.
- The expected RF gain is moderate, not near-perfect, because the task is now
  real temporal prediction rather than formula reconstruction.
- Data are aggregated by course, not by individual student.

See `docs/metodologia_modelagem_temporal.md` for the technical note intended for
SBIE/RBIE review.
