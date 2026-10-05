"""Testes do estimador de correlação intra-classe de `unidade_de_observacao.py`.

O erro que estes testes existem para impedir: um rho plausível mas errado é
indistinguível de um rho certo quando se olha só o dado real. A pendência 4.1
afirmava uma das duas pontas do intervalo como se fosse a resposta, e a única
defesa contra repetir isso é um estimador testado contra correlação **posta no
dado de propósito**.

Os três casos cobrem as três posições que importam: rho conhecido no meio, rho
igual a zero e rho igual a um. O quarto fixa a falha explícita em dado
desbalanceado, porque o estimador de ANOVA usado aqui só vale com m fixo e
devolver número em vez de erro seria o modo de falhar silencioso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

from unidade_de_observacao import icc_balanceado  # noqa: E402


def _painel(rho_alvo: float, k: int = 4000, m: int = 20, semente: int = 0):
    """Resíduo com correlação intra-classe conhecida, por decomposição.

    `r_ij = a_i + e_ij`, com `a_i` de variância rho e `e_ij` de variância
    1 - rho. A correlação entre duas linhas do mesmo grupo é então exatamente
    rho, porque elas compartilham `a_i` e nada mais.
    """
    rng = np.random.default_rng(semente)
    a = rng.normal(0.0, np.sqrt(rho_alvo), size=k)
    e = rng.normal(0.0, np.sqrt(1.0 - rho_alvo), size=(k, m))
    r = (a[:, None] + e).ravel()
    g = np.repeat(np.arange(k), m)
    return r, g


def test_recupera_rho_posto_no_dado():
    r, g = _painel(0.25)
    saida = icc_balanceado(r, g)
    assert abs(saida["rho"] - 0.25) < 0.02, saida["rho"]


def test_ruido_independente_da_rho_proximo_de_zero():
    rng = np.random.default_rng(1)
    k, m = 4000, 20
    r = rng.normal(size=k * m)
    g = np.repeat(np.arange(k), m)
    saida = icc_balanceado(r, g)
    assert abs(saida["rho"]) < 0.02, saida["rho"]
    # Sem correlação intragrupo, o efeito de desenho é 1 e o tamanho efetivo é
    # o número de LINHAS, não o de grupos. É a ponta que a pendência 4.1 negava.
    assert abs(saida["efeito_de_desenho"] - 1.0) < 0.05
    assert abs(saida["n_efetivo"] - k * m) / (k * m) < 0.05


def test_linhas_identicas_dentro_do_grupo_dao_rho_um():
    """A outra ponta: se as m linhas de um grupo são o mesmo número, o grupo
    carrega uma observação e o tamanho efetivo é o número de grupos."""
    rng = np.random.default_rng(2)
    k, m = 500, 20
    a = rng.normal(size=k)
    r = np.repeat(a, m)
    g = np.repeat(np.arange(k), m)
    saida = icc_balanceado(r, g)
    assert abs(saida["rho"] - 1.0) < 1e-9, saida["rho"]
    assert abs(saida["n_efetivo"] - k) / k < 1e-9
    assert abs(saida["razao_n_efetivo_sobre_n_celulas"] - 1.0) < 1e-9


def test_identidade_do_efeito_de_desenho():
    r, g = _painel(0.3, k=1000)
    s = icc_balanceado(r, g)
    assert abs(s["efeito_de_desenho"] - (1 + (s["m"] - 1) * s["rho"])) < 1e-12
    assert abs(s["n_efetivo"] - s["n_linhas"] / s["efeito_de_desenho"]) < 1e-9


def test_desbalanceado_falha_em_vez_de_devolver_numero():
    r = np.arange(9, dtype=float)
    g = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    with pytest.raises(ValueError):
        icc_balanceado(r, g)
