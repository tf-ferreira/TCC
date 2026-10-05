"""Testes das defasagens (seção 3 da especificação de atributos).

O teste central é um contraexemplo: um painel com buraco, em que defasar por
posição na série dá um número **errado e silencioso**, e defasar por calendário
dá ausência declarada. É a mesma família de armadilha de D9, preencher em vez de
declarar, e de D2, confundir existência de linha com presença.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "data"))

from defasagens import adicionar_defasagens, cobertura  # noqa: E402


def _painel_com_buraco() -> pd.DataFrame:
    """Loja 1, SKU 0, semanas 1, 2, 3 e depois 7 (faltam 4, 5 e 6)."""
    semanas = [1, 2, 3, 7]
    return pd.DataFrame({
        "store": [1] * 4,
        "sku": [0] * 4,
        "week": semanas,
        "preco_proprio": [4.0, 3.0, 5.0, 9.0],
        "alvo": [100.0, 180.0, 80.0, 40.0],
    })


def test_defasagem_de_uma_semana_pega_a_semana_certa():
    d = adicionar_defasagens(_painel_com_buraco(), ks_preco=(1,))
    esperado = [np.nan, 4.0, 3.0, np.nan]
    obtido = d["preco_lag1"].tolist()
    assert np.isnan(obtido[0]) and np.isnan(obtido[3])
    assert obtido[1:3] == esperado[1:3]


def test_o_buraco_produz_ausencia_e_nao_o_vizinho_errado():
    """Contraexemplo. A semana 7 não tem semana 6 no painel, então a defasagem
    não existe. Um `shift` por posição devolveria 5,0, o preço da semana 3, como
    se fossem semanas vizinhas. O erro não deixaria nenhuma linha vazia."""
    painel = _painel_com_buraco()
    d = adicionar_defasagens(painel, ks_preco=(1,))
    assert np.isnan(d.loc[d.week == 7, "preco_lag1"].iloc[0])

    por_posicao = painel.sort_values("week")["preco_proprio"].shift(1).tolist()
    assert por_posicao[3] == 5.0  # o que a implementação errada devolveria


def test_defasagem_de_quatro_semanas_atravessa_o_buraco_corretamente():
    """A semana 7 tem semana 3 no painel, então `lag4` existe e vale 5,0."""
    d = adicionar_defasagens(_painel_com_buraco(), ks_preco=(4,))
    assert d.loc[d.week == 7, "preco_lag4"].iloc[0] == 5.0


def test_defasagem_nao_cruza_series():
    """Duas lojas com as mesmas semanas: a defasagem de uma não pode vir da
    outra. É o mesmo cuidado que `fixar_categorias` protege no embedding."""
    painel = pd.DataFrame({
        "store": [1, 1, 2, 2],
        "sku": [0, 0, 0, 0],
        "week": [1, 2, 1, 2],
        "preco_proprio": [4.0, 3.0, 8.0, 7.0],
        "alvo": [10.0, 20.0, 30.0, 40.0],
    })
    d = adicionar_defasagens(painel, ks_preco=(1,))
    assert d.loc[(d.store == 1) & (d.week == 2), "preco_lag1"].iloc[0] == 4.0
    assert d.loc[(d.store == 2) & (d.week == 2), "preco_lag1"].iloc[0] == 8.0


def test_volume_defasado_sai_do_alvo_e_so_quando_pedido():
    painel = _painel_com_buraco()
    so_preco = adicionar_defasagens(painel, ks_preco=(1,))
    assert "volume_lag1" not in so_preco.columns

    com_volume = adicionar_defasagens(painel, ks_preco=(1,), ks_volume=(1,))
    assert com_volume.loc[com_volume.week == 2, "volume_lag1"].iloc[0] == 100.0


def test_o_numero_de_linhas_nao_muda():
    """A junção é à esquerda: defasar acrescenta colunas, nunca perde linha.
    Perder linha silenciosamente mudaria a amostra entre configurações e
    invalidaria a comparação preditiva."""
    painel = _painel_com_buraco()
    d = adicionar_defasagens(painel, ks_preco=tuple(range(1, 9)), ks_volume=(1,))
    assert len(d) == len(painel)


def test_cobertura_conta_o_que_esta_definido():
    d = adicionar_defasagens(_painel_com_buraco(), ks_preco=(1,))
    assert cobertura(d, ["preco_lag1"])["preco_lag1"] == 0.5
