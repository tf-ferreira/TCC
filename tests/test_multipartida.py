"""Testes da multipartida (D15 parte 2), antes de rodar com dado real."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

pytest.importorskip("torch")
import hierarquia as H  # noqa: E402
import multipartida as M  # noqa: E402

N = 6
PARES = [(1, 0), (3, 2)]


def _caixa(u0):
    return u0 + np.log(0.85), u0 + np.log(1.15)


def test_reparo_devolve_partidas_VIAVEIS():
    """A propriedade que justifica o reparo: sem ela toda partida inviável dispara
    o método de reserva, 22 vezes mais lento."""
    centro = np.zeros(N)
    rng = np.random.default_rng(0)
    u0 = rng.normal(0, 0.3, size=N)
    A, B = H.restricoes(PARES, centro, N, forma="nao_piorar", u0=u0[None, :])
    baixo, alto = _caixa(u0)
    U = rng.uniform(baixo, alto, size=(400, N))
    U2, ruim = M.reparar_para_viavel(U, A, B[0], baixo, alto)
    ok = ~ruim
    assert ok.sum() > 0
    assert np.all(U2[ok] @ A.T <= B[0][None, :] + 1e-12)
    assert np.all(U2 >= baixo - 1e-12) and np.all(U2 <= alto + 1e-12)


def test_reparo_nao_mexe_em_quem_ja_e_viavel():
    centro = np.zeros(N)
    u0 = np.zeros(N)
    A, B = H.restricoes(PARES, centro, N, forma="nao_piorar", u0=u0[None, :])
    baixo, alto = _caixa(u0)
    U = np.tile(u0, (5, 1))
    U2, ruim = M.reparar_para_viavel(U, A, B[0], baixo, alto)
    assert not ruim.any()
    assert np.allclose(U2, U)


def test_sem_restricao_o_reparo_so_recorta_na_caixa():
    u0 = np.zeros(N)
    baixo, alto = _caixa(u0)
    U = np.full((3, N), 10.0)
    U2, ruim = M.reparar_para_viavel(U, None, None, baixo, alto)
    assert not ruim.any()
    assert np.allclose(U2, alto)


def test_a_primeira_partida_e_SEMPRE_a_historica():
    """D15 pede que a histórica seja uma das K, e o relatório compara contra ela
    pela posição 0."""
    u0 = np.linspace(-0.2, 0.2, N)
    baixo, alto = _caixa(u0)
    U, _ = M.sortear_partidas(u0, baixo, alto, 10, semente=3)
    assert U.shape == (10, N)
    assert np.allclose(U[0], u0)


def test_sorteio_e_reprodutivel_e_muda_com_a_semente():
    u0 = np.zeros(N)
    baixo, alto = _caixa(u0)
    a, _ = M.sortear_partidas(u0, baixo, alto, 8, semente=1)
    b, _ = M.sortear_partidas(u0, baixo, alto, 8, semente=1)
    c, _ = M.sortear_partidas(u0, baixo, alto, 8, semente=2)
    assert np.allclose(a, b)
    assert not np.allclose(a[1:], c[1:])


def test_partidas_ficam_dentro_da_caixa():
    u0 = np.array([0.1, -0.3, 0.0, 0.25, -0.1, 0.05])
    baixo, alto = _caixa(u0)
    U, _ = M.sortear_partidas(u0, baixo, alto, 200, semente=5)
    assert np.all(U >= baixo - 1e-12) and np.all(U <= alto + 1e-12)


def test_k_igual_a_um_da_so_a_historica():
    u0 = np.zeros(N)
    baixo, alto = _caixa(u0)
    U, ruim = M.sortear_partidas(u0, baixo, alto, 1, semente=0)
    assert U.shape == (1, N) and ruim == 0
