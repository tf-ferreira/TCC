"""Testes de `suavidade_d16.py`, sem torch.

A propriedade central: com pesos quaisquer (inclusive muito negativos antes do
softplus), `mᵢ″ > 0`, logo a elasticidade própria é monótona e não oscila. O
teste sorteia pesos agressivos de propósito, porque a afirmação é "para qualquer
valor dos pesos", e um teste com pesos bem comportados não a provaria.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "experiments"))

from suavidade_d16 import analisar, derivadas, trocas_de_sinal, u_de_elasticidade_um  # noqa: E402


def _pesos(semente, n=5, h=8, escala=4.0):
    rng = np.random.default_rng(semente)
    return (rng.normal(0, escala, (n, h)), rng.normal(0, escala, (n, h)),
            rng.normal(0, escala, (n, h)))


def test_segunda_derivada_positiva_para_pesos_quaisquer():
    u = np.linspace(-3, 3, 4001)
    for s in range(20):
        _, m2 = derivadas(*_pesos(s), u)
        assert m2.min() > 0


def test_forma_fechada_bate_com_diferenca_finita():
    w1, w2, b = _pesos(1, escala=1.0)
    u = np.linspace(-1, 1, 201)
    h = 1e-5
    m1, m2 = derivadas(w1, w2, b, u)
    m1p, _ = derivadas(w1, w2, b, u + h)
    m1m, _ = derivadas(w1, w2, b, u - h)
    assert np.allclose((m1p - m1m) / (2 * h), m2, atol=1e-5)

    def m(x):
        z = x[:, None, None] * np.logaddexp(0, w1) + b
        return (np.logaddexp(0, z) * np.logaddexp(0, w2)).sum(-1)
    assert np.allclose((m(u + h) - m(u - h)) / (2 * h), m1, atol=1e-5)


def test_trocas_de_sinal_conta_certo():
    assert trocas_de_sinal(np.array([1, 2, -1, -3, 4])) == 2
    assert trocas_de_sinal(np.array([1, 0, 1])) == 0


def test_bissecao_acha_elasticidade_um():
    w1, w2, b = _pesos(2, escala=1.0)
    raiz, lo, hi = u_de_elasticidade_um(w1, w2, b, -10, 10)
    ok = np.isfinite(raiz)
    assert ok.any()
    m1, _ = derivadas(w1, w2, b, raiz[None, ok].repeat(1, 0))
    # derivadas aceita (P, n) só com n colunas; avaliar SKU a SKU
    for i in np.where(ok)[0]:
        v = derivadas(w1[i:i + 1], w2[i:i + 1], b[i:i + 1], np.array([raiz[i]]))[0][0, 0]
        assert abs(v - 1) < 1e-8


def test_terceira_derivada_troca_de_sinal_e_isso_nao_e_oscilacao():
    """Registro do erro da primeira versão: contar trocas da SEGUNDA diferença de ε
    acusava 'oscilação' em pesos corretos, porque ela mede −m‴."""
    w1, w2, b = _pesos(3, escala=1.0)
    u = np.linspace(-3, 3, 2001)
    eps = -derivadas(w1, w2, b, u)[0]
    assert sum(trocas_de_sinal(np.diff(eps[:, i], 2), 1e-12) for i in range(5)) > 0
    assert sum(trocas_de_sinal(np.diff(eps[:, i]), 1e-14) for i in range(5)) == 0


def test_analisar_nao_acha_oscilacao_e_amplitude_positiva():
    w1, w2, b = _pesos(3, escala=1.0)
    U = np.random.default_rng(0).normal(0, 0.2, (500, 5))
    r = analisar(w1, w2, b, U)
    assert r["total_trocas_de_sinal"] == 0 and r["min_m2_global"] > 0
    assert r["amplitude_por_celula"]["mediana"] > 0
