"""Testes dos estimadores de `elasticidades.py`.

Todos usam dados gerados com parâmetro conhecido: o teste pergunta se o
estimador devolve o número que foi posto lá dentro. É a única forma de
separar "o código roda" de "o código estima o que diz estimar", e o projeto
já pagou o preço de não ter isso, com uma elasticidade publicada sem produtor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from elasticidades import centralizar, cragg_donald, ols, tsls


def test_centralizar_absorve_dois_efeitos_fixos_exatamente():
    """Com efeitos fixos aditivos e sem ruído, o coeficiente tem que voltar
    exato. Se a centralização alternada não convergisse, voltaria enviesado."""
    rng = np.random.default_rng(7)
    n_i, n_t = 12, 9
    i = np.repeat(np.arange(n_i), n_t)
    t = np.tile(np.arange(n_t), n_i)
    alfa = rng.normal(size=n_i)[i]
    gama = rng.normal(size=n_t)[t]
    x = rng.normal(size=n_i * n_t)
    beta = 2.0
    y = alfa + gama + beta * x

    M = centralizar(np.column_stack([y, x]), [i, t])
    b, _, _ = ols(M[:, 0], M[:, 1:2], k=1 + n_i + n_t)
    assert abs(float(b[0]) - beta) < 1e-8


def test_centralizar_e_idempotente():
    rng = np.random.default_rng(3)
    i = np.repeat(np.arange(6), 5)
    t = np.tile(np.arange(5), 6)
    M = rng.normal(size=(30, 2))
    uma = centralizar(M, [i, t])
    duas = centralizar(uma, [i, t])
    assert np.max(np.abs(uma - duas)) < 1e-10


def _endogeno(n: int, beta: float, forca: float, semente: int):
    """x correlacionado com o erro; z desloca x e não entra em y."""
    rng = np.random.default_rng(semente)
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    x = forca * z + u + 0.3 * rng.normal(size=n)
    y = beta * x + u
    return y, x[:, None], z[:, None]


def test_mqo_e_enviesado_e_iv_recupera_o_parametro():
    beta = -2.0
    y, X, Z = _endogeno(20_000, beta, forca=1.0, semente=11)
    b_ols, _, _ = ols(y, X, k=1)
    b_iv, ep, _ = tsls(y, X, Z, k=1)
    # O erro entra em x com sinal positivo, então MQO puxa o coeficiente
    # para cima: menos negativo que a verdade.
    assert float(b_ols[0]) > beta + 0.2
    assert abs(float(b_iv[0]) - beta) < 0.05
    assert abs(float(b_iv[0]) - beta) < 4 * float(ep[0])


def test_erro_padrao_de_iv_cai_com_a_raiz_de_n():
    beta = -2.0
    _, ep1, _ = tsls(*_endogeno(5_000, beta, 1.0, 5), k=1)
    _, ep2, _ = tsls(*_endogeno(20_000, beta, 1.0, 5), k=1)
    razao = float(ep1[0]) / float(ep2[0])
    assert 1.7 < razao < 2.3


def test_cragg_donald_cresce_com_a_forca_do_instrumento():
    _, x_fraco, z_fraco = _endogeno(5_000, -2.0, forca=0.05, semente=2)
    _, x_forte, z_forte = _endogeno(5_000, -2.0, forca=1.00, semente=2)
    fraco = cragg_donald(x_fraco, z_fraco, k_fe=1)
    forte = cragg_donald(x_forte, z_forte, k_fe=1)
    assert fraco > 0
    assert forte > 50 * fraco
