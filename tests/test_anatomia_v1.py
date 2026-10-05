"""Testes da anatomia da política da v1 (`src/optimization/anatomia_v1.py`).

Só as funções puras: a direção da variação de preço, o teto do treino e a
fração acima dele. O laço de sementes reusa funções já testadas em
`test_contrafactual_item9.py` e `test_mundos.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "optimization"))

import anatomia_v1 as AN  # noqa: E402


def test_direcao_conta_subida_descida_e_bordas():
    b, a = AN.LOG_BAIXO, AN.LOG_ALTO
    du = np.array([[b, 0.0, a],
                   [0.05, -0.05, a]])
    d = AN.direcao(du)
    assert d["sobe_por_sku"] == [0.5, 0.0, 1.0]
    assert d["desce_por_sku"] == [0.5, 0.5, 0.0]
    assert d["borda_baixa_por_sku"] == [0.5, 0.0, 0.0]
    assert d["borda_alta_por_sku"] == [0.0, 0.0, 1.0]
    assert d["sobe"] == pytest.approx(3 / 6)
    assert d["desce"] == pytest.approx(2 / 6)
    assert d["interior"] == pytest.approx(1 - 1 / 6 - 2 / 6)
    assert d["maior_saida_da_caixa"] == 0.0


def test_direcao_registra_saida_da_caixa_sem_levantar():
    du = np.array([[AN.LOG_ALTO + 0.01, 0.0]])
    d = AN.direcao(du)
    assert d["maior_saida_da_caixa"] == pytest.approx(0.01)


def test_direcao_exige_matriz():
    with pytest.raises(ValueError):
        AN.direcao(np.zeros(3))


def test_teto_do_treino_por_loja_e_reserva_para_loja_nova():
    cel = pd.DataFrame({"store": [1, 1, 2, 2], "week": [1, 2, 1, 50],
                        "preco_00": [2.0, 2.5, 3.0, 9.0],
                        "preco_01": [1.0, 0.0, 1.5, 9.0]})
    chaves = pd.DataFrame({"store": [1, 2, 3], "week": [60, 60, 60]})
    t = AN.tetos_do_treino(cel, 2, ultima_semana=10, chaves=chaves)
    # a semana 50 fica fora do treino; preço zero não é preço
    np.testing.assert_allclose(t[0], [2.5, 1.0])
    np.testing.assert_allclose(t[1], [3.0, 1.5])
    # a loja 3 não tem treino: recebe o maior do SKU entre as lojas do treino
    np.testing.assert_allclose(t[2], [3.0, 1.5])


def test_acima_do_teto():
    p = np.array([[1.0, 2.01], [1.2, 2.0]])
    teto = np.array([[1.1, 2.0], [1.1, 2.0]])
    r = AN.acima_do_teto(p, teto)
    assert r["por_sku"] == [0.5, 0.5]
    assert r["total"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        AN.acima_do_teto(p, teto[:1])
