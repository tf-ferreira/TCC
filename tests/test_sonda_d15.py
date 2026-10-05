"""Testes da sonda de D15: a codificação do ReLU tem de ser EXATA.

O risco aqui não é o código quebrar, é ele resolver **outro** problema: uma
codificação frouxa devolve valor maior que o ótimo verdadeiro e o "certificado" passa
a certificar coisa errada. Por isso os testes comparam o SCIP contra busca em grade
fina no mesmo modelo, e não contra si mesmo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for sub in ("models", "data", "experiments", "optimization"):
    sys.path.insert(0, str(RAIZ / "src" / sub))

pytest.importorskip("pyscipopt")
import sonda_d15 as S  # noqa: E402


def _pesos(n=2, h=3, semente=0):
    """Rede de brinquedo com pesos conhecidos e nenhum contexto."""
    rng = np.random.default_rng(semente)
    return {"n_prod": n, "h": h, "escala": np.ones(n),
            "W1": rng.normal(0, 1.5, size=(h, n)),
            "b1": rng.normal(0, 0.5, size=h),
            "W2": rng.normal(0, 1.5, size=(n, h)),
            "b2": rng.uniform(5, 15, size=n),
            "rmse_teste": 0.0}


def test_limites_da_preativacao_batem_com_os_CANTOS():
    """Aritmética de intervalo numa camada linear é exata, e o extremo está num
    canto da caixa. Se não batesse, o big-M ficaria frouxo ou ERRADO."""
    rng = np.random.default_rng(1)
    W, b = rng.normal(size=(4, 3)), rng.normal(size=4)
    lo, hi = np.array([0.5, 1.0, 2.0]), np.array([1.5, 2.0, 2.5])
    z_lo, z_hi = S.limites_das_preativacoes(W, b, lo, hi)
    cantos = np.array(np.meshgrid(*[[lo[j], hi[j]] for j in range(3)])).reshape(3, -1).T
    vals = cantos @ W.T + b
    assert np.allclose(z_lo, vals.min(axis=0))
    assert np.allclose(z_hi, vals.max(axis=0))


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_o_global_do_SCIP_bate_com_busca_em_grade(semente):
    pesos = _pesos(n=2, h=4, semente=semente)
    ctx = np.zeros(0)
    p0 = np.array([1.0, 2.0])
    r = S.resolver_global(pesos, ctx, p0, limite_s=60)
    assert r["fechou"], r["status"]

    W1p, b1e = S.dobrar_contexto(pesos, ctx)
    lo, hi = 0.85 * p0, 1.15 * p0
    g = [np.linspace(lo[i], hi[i], 241) for i in range(2)]
    melhor = max(S.receita(pesos, W1p, b1e, np.array([x, y]))
                 for x in g[0] for y in g[1])
    # A grade nao contem o otimo exato, entao ela e um LIMITE INFERIOR: o SCIP tem
    # de ser maior ou igual, e nao muito maior.
    assert r["valor"] >= melhor - 1e-6
    assert r["valor"] <= melhor * (1 + 5e-3)


def test_o_ponto_devolvido_e_viavel_e_reproduz_o_valor():
    """Sem isso, um valor certo com ponto errado passaria: o teste liga os dois."""
    pesos = _pesos(n=3, h=5, semente=3)
    ctx = np.zeros(0)
    p0 = np.array([1.0, 1.5, 0.8])
    r = S.resolver_global(pesos, ctx, p0, limite_s=60)
    assert r["fechou"]
    p = np.array(r["p"])
    # A tolerancia e 1e-6 porque e a `feastol` padrao do SCIP: ele entrega o ponto
    # na borda com esse residuo, e exigir 1e-9 seria exigir do solver precisao que
    # ele nao promete. Mesma logica da tolerancia declarada em `problema.py`.
    folga = 1e-6 * np.maximum(1.0, np.abs(p0))
    assert np.all(p >= 0.85 * p0 - folga) and np.all(p <= 1.15 * p0 + folga)
    W1p, b1e = S.dobrar_contexto(pesos, ctx)
    assert S.receita(pesos, W1p, b1e, p) == pytest.approx(r["valor"], rel=1e-6)


def test_o_aproximado_nunca_passa_do_global():
    """Relação de ordem obrigatória: o ótimo global é limite superior de qualquer
    método. Se o aproximado passar, a codificação está frouxa."""
    pesos = _pesos(n=3, h=6, semente=4)
    ctx = np.zeros(0)
    p0 = np.array([1.2, 0.9, 1.7])
    g = S.resolver_global(pesos, ctx, p0, limite_s=60)
    a = S.resolver_aproximado(pesos, ctx, p0, k=30)
    assert g["fechou"]
    assert a["melhor"] <= g["valor"] * (1 + 1e-6)


def test_gradiente_do_aproximado_bate_com_diferenca_finita():
    pesos = _pesos(n=3, h=6, semente=5)
    ctx = np.zeros(0)
    W1p, b1e = S.dobrar_contexto(pesos, ctx)
    p = np.array([1.1, 1.3, 0.95])
    # o gradiente vive dentro de resolver_aproximado; aqui a conta equivalente
    z1 = W1p @ (p / pesos["escala"]) + b1e
    a1 = np.maximum(z1, 0.0)
    z2 = pesos["W2"] @ a1 + pesos["b2"]
    v = np.maximum(z2, 0.0)
    J = ((z2 > 0)[:, None] * (pesos["W2"] @ ((z1 > 0)[:, None] * W1p / pesos["escala"])))
    grad = v + J.T @ p
    h = 1e-7
    fd = np.array([(S.receita(pesos, W1p, b1e, p + h * np.eye(3)[i])
                    - S.receita(pesos, W1p, b1e, p - h * np.eye(3)[i])) / (2 * h)
                   for i in range(3)])
    assert np.allclose(grad, fd, atol=1e-4)


def test_relus_fixos_sao_contados_e_a_caixa_estreita_fixa_mais():
    """O diagnóstico que explica o tempo: caixa mais estreita fixa mais ReLU, e
    ReLU fixo o presolve elimina."""
    pesos = _pesos(n=3, h=8, semente=6)
    ctx = np.zeros(0)
    p0 = np.array([1.0, 1.0, 1.0])
    estreita = S.resolver_global(pesos, ctx, p0, limites=(0.99, 1.01), limite_s=60)
    larga = S.resolver_global(pesos, ctx, p0, limites=(0.10, 3.00), limite_s=60)
    assert estreita["pct_relus_fixos_na_caixa"] >= larga["pct_relus_fixos_na_caixa"]
    assert estreita["pct_relus_fixos_na_caixa"] == 100.0
