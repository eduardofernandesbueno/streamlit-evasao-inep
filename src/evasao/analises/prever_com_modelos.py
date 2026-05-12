from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CAMINHO_MODELO = ROOT / "modelos" / "modelos_salvos" / "modelo_melhor_evasao.pkl"

FEATURE_COLS = [
    "ingressantes_t",
    "concluintes_t",
    "matriculados_t",
    "vagas_totais_t",
    "inscritos_totais_t",
]


def main() -> None:
    """
    Executa uma predicao pontual com o modelo temporal atual.

    Este script e apenas um exemplo auxiliar. Para o artigo SBIE/RBIE, o centro
    metodologico permanece em src/evasao/modelagem/randomforest.py.
    """
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo temporal nao encontrado. Execute "
            "`python -m src.evasao.modelagem.randomforest` antes deste exemplo."
        )

    modelo = joblib.load(CAMINHO_MODELO)

    # Exemplo com variaveis observadas no ano-base t. Nao usar taxas
    # contemporaneas de conclusao/evasao como features preditivas principais.
    entrada = pd.DataFrame(
        [
            {
                "ingressantes_t": 120.0,
                "concluintes_t": 65.0,
                "matriculados_t": 430.0,
                "vagas_totais_t": 160.0,
                "inscritos_totais_t": 980.0,
            }
        ],
        columns=FEATURE_COLS,
    )

    predicao = float(modelo.predict(entrada)[0])
    print(f"Predicao temporal de taxa_evasao_alvo em t+1: {predicao:.4f}")


if __name__ == "__main__":
    main()
