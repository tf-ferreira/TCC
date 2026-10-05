"""Testes de `espelho_d16.py`, sem torch: o comparador recebe as previsões prontas.

O que precisa ser verdade: uma "rede" com a elasticidade certa devolve diferença
perto de zero, uma com a elasticidade errada devolve a distância certa, e a
diferença estimada direto é exatamente β_obs − β_esp (2SLS é linear em y).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for pasta in ("data", "experiments"):
    sys.path.insert(0, str(RAIZ / "src" / pasta))

from espelho_d16 import comparar_espelho  # noqa: E402

EPS = -2.0


def _dado(semente=0, S=50, N=3, T=120):
    rng = np.random.default_rng(semente)
    loja = np.repeat(np.arange(S), N * T)
    sku = np.tile(np.repeat(np.arange(N), T), S)
    semana = np.tile(np.arange(T), S * N)
    n = len(loja)
    serie = loja * 100 + sku
    fe = rng.normal(0, 1, S * 100)[serie] + rng.normal(0, 1, T)[semana]
    lnc = rng.normal(0, 0.1, n)
    choque = rng.normal(0, 0.1, n)
    lnp = 0.8 * lnc + 0.5 * choque + rng.normal(0, 0.05, n)
    y = fe + EPS * lnp + choque + rng.normal(0, 0.05, n)
    return y, lnp, lnc, serie, semana, loja, fe


def test_rede_certa_da_diferenca_perto_de_zero_e_rede_errada_da_a_distancia():
    y, lnp, lnc, serie, semana, loja, fe = _dado()
    certa = fe + EPS * lnp          # superfície estrutural, sem o choque
    errada = fe - 1.0 * lnp
    r = comparar_espelho(y, np.column_stack([certa, errada]), lnp, lnc,
                         serie, semana, loja)
    assert abs(r["obs"]["coef"] - EPS) < 0.08
    assert abs(r["por_semente"][0]["diferenca"]["coef"]) < 0.08
    assert abs(r["por_semente"][1]["diferenca"]["coef"] - (-1.0)) < 0.08


def test_diferenca_direta_e_exatamente_a_subtracao():
    y, lnp, lnc, serie, semana, loja, fe = _dado(semente=1)
    rng = np.random.default_rng(3)
    H = np.column_stack([fe - 1.7 * lnp + rng.normal(0, 0.1, len(y)) for _ in range(3)])
    r = comparar_espelho(y, H, lnp, lnc, serie, semana, loja)
    for p in r["por_semente"]:
        assert abs(p["diferenca"]["coef"]
                   - (r["obs"]["coef"] - p["espelho"]["coef"])) < 1e-9


def test_rede_que_copia_o_dado_empata_mesmo_com_o_choque():
    """O aviso do docstring: em treino, ajuste perfeito empata sem estrutura."""
    y, lnp, lnc, serie, semana, loja, _ = _dado(semente=2)
    r = comparar_espelho(y, y[:, None], lnp, lnc, serie, semana, loja)
    assert abs(r["por_semente"][0]["diferenca"]["coef"]) < 1e-9
