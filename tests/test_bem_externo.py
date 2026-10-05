"""Testes do índice do bem externo (D23).

O índice é um Laspeyres de preços relativos com pesos de receita da janela
base, normalizado pelos pesos efetivamente presentes na célula. Os testes
fixam as três propriedades das quais o argumento de D23 depende: os pesos
somam 1, o índice vale 1 no preço base, e a normalização impede que a
ausência de um SKU na célula desloque o índice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

from bem_externo import agregar, cesta_base

BASE_ATE = 41  # semanas 1..40 formam a janela base


def _resto(linhas: list[dict]) -> pd.DataFrame:
    padrao = {"store": 1, "week": 1, "upc": 900, "preco": 1.0, "custo": 0.5,
              "volume": 10.0, "promo": 0.0}
    return pd.DataFrame([{**padrao, **l} for l in linhas])


def _janela_base() -> list[dict]:
    """Dois SKUs com 40 semanas cada: A a US$ 1 e B a US$ 4, mesmo volume.

    Receita da base: A = 40 x 1 x 10 = 400; B = 40 x 4 x 10 = 1600. Logo os
    pesos são 0,2 e 0,8, calculáveis à mão.
    """
    return [{"upc": upc, "week": w, "preco": p, "volume": 10.0}
            for upc, p in ((901, 1.0), (902, 4.0))
            for w in range(1, BASE_ATE)]


def test_pesos_somam_um_e_sao_de_receita():
    ref = cesta_base(_resto(_janela_base()), BASE_ATE)
    assert abs(float(ref.w.sum()) - 1.0) < 1e-12
    assert abs(float(ref.loc[901, "w"]) - 0.2) < 1e-12
    assert abs(float(ref.loc[902, "w"]) - 0.8) < 1e-12


def test_sku_com_poucas_observacoes_fica_de_fora():
    linhas = _janela_base() + [{"upc": 903, "week": w, "preco": 9.0}
                               for w in range(1, 30)]  # 29 < 30 exigidas
    ref = cesta_base(_resto(linhas), BASE_ATE)
    assert set(ref.index) == {901, 902}


def test_truncar_peso_reduz_a_concentracao():
    ref = cesta_base(_resto(_janela_base()), BASE_ATE)
    truncada = cesta_base(_resto(_janela_base()), BASE_ATE, truncar_peso=50)
    assert float(truncada.w.max()) < float(ref.w.max())
    assert abs(float(truncada.w.sum()) - 1.0) < 1e-12


def test_indice_vale_um_no_preco_base_e_dois_quando_tudo_dobra():
    linhas = _janela_base()
    linhas += [{"upc": 901, "week": 41, "preco": 1.0},
               {"upc": 902, "week": 41, "preco": 4.0},
               {"upc": 901, "week": 42, "preco": 2.0},
               {"upc": 902, "week": 42, "preco": 8.0}]
    d = _resto(linhas)
    fora = agregar(d, cesta_base(d, BASE_ATE)).set_index("week")
    assert abs(float(fora.loc[41, "idx_preco_resto"]) - 1.0) < 1e-12
    assert abs(float(fora.loc[42, "idx_preco_resto"]) - 2.0) < 1e-12


def test_ausencia_de_sku_na_celula_nao_desloca_o_indice():
    """Se o SKU pesado falta na célula, o índice tem que refletir só quem
    está presente, e não cair por falta de peso. É o que a divisão por
    `soma_w` garante, e é o que permite a D23 definir o agregado sobre quem
    estiver presente sem reintroduzir a exigência de completude de D12."""
    linhas = _janela_base()
    linhas += [{"upc": 901, "week": 43, "preco": 3.0}]  # só o SKU leve, a 3x
    d = _resto(linhas)
    fora = agregar(d, cesta_base(d, BASE_ATE)).set_index("week")
    assert abs(float(fora.loc[43, "idx_preco_resto"]) - 3.0) < 1e-12
    assert int(fora.loc[43, "n_resto"]) == 1


def test_indice_de_custo_ignora_apenas_o_sku_sem_custo():
    linhas = _janela_base()
    linhas += [{"upc": 901, "week": 44, "preco": 2.0, "custo": 1.0},
               {"upc": 902, "week": 44, "preco": 4.0, "custo": np.nan}]
    d = _resto(linhas)
    fora = agregar(d, cesta_base(d, BASE_ATE)).set_index("week")
    # preço: 0,2 x 2 + 0,8 x 1 = 1,2 sobre peso 1,0
    assert abs(float(fora.loc[44, "idx_preco_resto"]) - 1.2) < 1e-12
    # custo: só o 901, a 1,0 contra base 0,5, logo 2,0
    assert abs(float(fora.loc[44, "idx_custo_resto"]) - 2.0) < 1e-12
